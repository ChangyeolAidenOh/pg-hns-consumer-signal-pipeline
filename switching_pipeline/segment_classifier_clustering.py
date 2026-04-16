import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

VOC_FEATURES_PATH = "switching_pipeline/data/voc_features.csv"
TREND_FEATURES_PATH = "switching_pipeline/data/trend_features_processed.csv"
OUTPUT_DIR = "switching_pipeline/data"


def load_features():
    voc = pd.read_csv(VOC_FEATURES_PATH)
    trend = pd.read_csv(TREND_FEATURES_PATH, index_col=0, parse_dates=True)
    return voc, trend


def build_cluster_features(voc_df):
    """
    Select features for unsupervised clustering.
    Includes all behavioral signals without segment labels.
    Goal: discover natural groupings in consumer behavior
    without imposing predefined segment definitions.
    """
    feature_cols = [
        "is_youtube", "is_blog", "is_cafe",
        "is_churn", "is_positive",
        "is_medical_frame", "is_ingredient_frame", "is_comparison_frame",
        "is_competitor", "churn_score", "positive_score"
    ]
    X = voc_df[feature_cols].values
    return X, feature_cols


def find_optimal_k(X_scaled, k_range=range(2, 8)):
    """
    Find optimal number of clusters using silhouette score.
    Silhouette score measures how similar a point is to its own cluster
    vs other clusters. Higher is better (-1 to 1 scale).
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"  Optimal k: {best_k} (silhouette={scores[best_k]:.4f})")
    return best_k


def interpret_cluster(cluster_df, feature_cols):
    """
    Interpret cluster characteristics based on feature means.
    Maps discovered clusters to consumer behavior profiles.
    """
    means = cluster_df[feature_cols].mean()

    # scoring criteria for segment interpretation
    churn_signal = means["is_churn"] + means["churn_score"] * 0.1
    positive_signal = means["is_positive"] + means["positive_score"] * 0.1
    competitor_signal = means["is_competitor"] + means["is_comparison_frame"]
    medical_signal = means["is_medical_frame"] + means["is_ingredient_frame"]

    if competitor_signal > 0.3 and churn_signal > 0.5:
        return "Active Switcher"
    elif medical_signal > 0.3 and churn_signal > 0.5:
        return "At-risk"
    elif positive_signal > churn_signal:
        return "Loyal"
    else:
        return "Passive User"


def compute_switching_probability_cluster(voc_df, trend_df, cluster_col="cluster_segment"):
    """
    Compute switching probability per discovered cluster.
    Uses same trend adjustment as regression approach for comparison.
    Base probability derived from cluster churn density
    rather than logistic regression output.
    """
    latest = trend_df.iloc[-1]
    antitro_pressure = min(latest["antitro_ratio"] / 2.0, 1.0)
    hns_decline = max(-latest["hns_momentum"], 0)
    hns_decline = min(hns_decline, 1.0)
    trend_multiplier = 1.0 + 0.3 * antitro_pressure + 0.2 * hns_decline

    seg_stats = voc_df.groupby(cluster_col).agg(
        n_docs=("is_churn", "count"),
        churn_rate=("is_churn", "mean"),
        competitor_rate=("is_competitor", "mean"),
        medical_rate=("is_medical_frame", "mean"),
        ingredient_rate=("is_ingredient_frame", "mean"),
        mean_churn_score=("churn_score", "mean")
    ).reset_index()

    # base probability from churn density
    seg_stats["base_prob"] = (
        seg_stats["churn_rate"] * 0.6 +
        seg_stats["competitor_rate"] * 0.4
    ).round(4)

    seg_stats["switching_probability"] = (
        seg_stats["base_prob"] * trend_multiplier
    ).clip(upper=0.95).round(4)

    return seg_stats, trend_multiplier


if __name__ == "__main__":
    voc_df, trend_df = load_features()
    print(f"VoC features: {len(voc_df)} | Trend features: {len(trend_df)}")

    X, feature_cols = build_cluster_features(voc_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # find optimal k
    print("\nSilhouette Score by k:")
    best_k = find_optimal_k(X_scaled)

    # fit KMeans
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    voc_df["cluster_id"] = cluster_labels

    # interpret clusters
    print(f"\nCluster Profiles (k={best_k}):")
    cluster_names = {}
    for cid in range(best_k):
        sub = voc_df[voc_df["cluster_id"] == cid]
        name = interpret_cluster(sub, feature_cols)
        cluster_names[cid] = name
        print(f"\n  Cluster {cid} → {name} (n={len(sub)})")
        means = sub[feature_cols].mean().round(3)
        print(f"    churn={means['is_churn']:.3f} | "
              f"positive={means['is_positive']:.3f} | "
              f"medical={means['is_medical_frame']:.3f} | "
              f"ingredient={means['is_ingredient_frame']:.3f} | "
              f"competitor={means['is_competitor']:.3f}")

    voc_df["cluster_segment"] = voc_df["cluster_id"].map(cluster_names)

    # compare with rule-based segments
    print("\nCluster vs Rule-based Segment Comparison:")
    print(pd.crosstab(voc_df["cluster_segment"],
                      voc_df["segment"]).to_string())

    # switching probability
    print(f"\nSwitching Probability per Cluster")
    seg_prob, trend_mult = compute_switching_probability_cluster(voc_df, trend_df)
    print(f"  Trend multiplier: {trend_mult:.3f}")
    print(seg_prob[["cluster_segment", "n_docs", "churn_rate",
                    "competitor_rate", "switching_probability"]].to_string(index=False))

    # save
    voc_df.to_csv(f"{OUTPUT_DIR}/voc_features_clustered.csv",
                  index=False, encoding="utf-8-sig")
    seg_prob.to_csv(f"{OUTPUT_DIR}/switching_prob_clustering.csv",
                    index=False, encoding="utf-8-sig")
    print(f"\n switching_prob_clustering.csv")