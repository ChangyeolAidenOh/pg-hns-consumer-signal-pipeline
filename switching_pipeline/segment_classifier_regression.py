import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

VOC_FEATURES_PATH = "switching_pipeline/data/voc_features.csv"
TREND_FEATURES_PATH = "switching_pipeline/data/trend_features_processed.csv"
OUTPUT_DIR = "switching_pipeline/data"


def load_features():
    voc = pd.read_csv(VOC_FEATURES_PATH)
    trend = pd.read_csv(TREND_FEATURES_PATH, index_col=0, parse_dates=True)
    return voc, trend


def build_model_features(voc_df):
    """
    Select features for logistic regression.
    Excludes rule-defining features (is_churn, is_competitor, is_comparison_frame)
    to avoid data leakage — these are used to define segments, not predict them.
    Uses behavioral proxies instead:
    - channel (youtube/blog/cafe)
    - frame type (medical, ingredient)
    - signal intensity (churn_score, positive_score)
    """
    feature_cols = [
        "is_youtube", "is_blog", "is_cafe",
        "is_medical_frame", "is_ingredient_frame",
        "churn_score", "positive_score"
    ]
    X = voc_df[feature_cols].values
    y = voc_df["segment"].values
    return X, y, feature_cols


def find_optimal_regularization(X_scaled, y_encoded):
    """
    Grid search over regularization parameter C.
    Lower C = stronger regularization = simpler model.
    Higher C = weaker regularization = fits data more closely.
    Optimal C selected by StratifiedKFold cross-validation accuracy.
    """
    param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}
    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring="accuracy")
    grid.fit(X_scaled, y_encoded)

    print("  Regularization search results:")
    for mean, std, params in zip(
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"],
        grid.cv_results_["params"]
    ):
        print(f"    C={params['C']:.3f}: {mean:.4f} ± {std:.4f}")
    print(f"  Optimal C: {grid.best_params_['C']} "
          f"(accuracy={grid.best_score_:.4f})")
    return grid.best_estimator_, grid.best_params_["C"]


def find_optimal_trend_weights(voc_df, trend_df, clf, scaler, le, feature_cols):
    """
    Grid search over trend multiplier weights (w_antitro, w_hns).
    Objective: maximize separation between Active Switcher
    and Loyal switching probabilities.
    Larger gap = more discriminative probability output.
    Weights are data-driven rather than arbitrarily assigned.
    """
    latest = trend_df.iloc[-1]
    antitro_pressure = min(latest["antitro_ratio"] / 2.0, 1.0)
    hns_decline = max(-latest["hns_momentum"], 0)
    hns_decline = min(hns_decline, 1.0)

    switcher_idx = list(le.classes_).index("Active Switcher")

    # precompute base probabilities per segment
    base_probs = {}
    for seg in voc_df["segment"].unique():
        sub = voc_df[voc_df["segment"] == seg]
        X_sub = scaler.transform(sub[feature_cols].values)
        base_probs[seg] = clf.predict_proba(X_sub)[:, switcher_idx].mean()

    # grid search over weight combinations
    best_gap = -1
    best_weights = (0.3, 0.2)
    results = []

    for w1 in np.arange(0.1, 0.6, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            multiplier = 1.0 + w1 * antitro_pressure + w2 * hns_decline
            switcher_prob = min(base_probs.get("Active Switcher", 0) * multiplier, 0.95)
            loyal_prob = min(base_probs.get("Loyal", 0) * multiplier, 0.95)
            gap = switcher_prob - loyal_prob
            results.append({
                "w_antitro": round(w1, 2),
                "w_hns": round(w2, 2),
                "multiplier": round(multiplier, 3),
                "switcher_prob": round(switcher_prob, 4),
                "loyal_prob": round(loyal_prob, 4),
                "gap": round(gap, 4)
            })
            if gap > best_gap:
                best_gap = gap
                best_weights = (round(w1, 2), round(w2, 2))

    results_df = pd.DataFrame(results).sort_values("gap", ascending=False)
    print("\n  Top 5 weight combinations (by prob gap):")
    print(results_df.head(5).to_string(index=False))
    print(f"\n  Optimal weights: w_antitro={best_weights[0]:.2f}, "
          f"w_hns={best_weights[1]:.2f} (gap={best_gap:.4f})")
    return best_weights


def compute_final_switching_probability(voc_df, trend_df, clf, scaler,
                                        le, feature_cols, best_w1, best_w2):
    """
    Compute final switching probability per segment using:
    - Logistic regression base probability (Layer 1 VoC signal)
    - Trend multiplier with data-driven optimal weights (Layer 2)
    Final probability = base_prob * trend_multiplier, capped at 0.95.
    """
    latest = trend_df.iloc[-1]
    antitro_pressure = min(latest["antitro_ratio"] / 2.0, 1.0)
    hns_decline = max(-latest["hns_momentum"], 0)
    hns_decline = min(hns_decline, 1.0)
    trend_multiplier = 1.0 + best_w1 * antitro_pressure + best_w2 * hns_decline

    print(f"  Trend multiplier: {trend_multiplier:.3f} "
          f"(w_antitro={best_w1}, w_hns={best_w2})")

    switcher_idx = list(le.classes_).index("Active Switcher")
    seg_results = []

    for seg in ["Active Switcher", "At-risk", "Passive User", "Loyal"]:
        sub = voc_df[voc_df["segment"] == seg]
        if len(sub) == 0:
            continue
        X_sub = scaler.transform(sub[feature_cols].values)
        base_prob = clf.predict_proba(X_sub)[:, switcher_idx].mean()
        switching_prob = min(base_prob * trend_multiplier, 0.95)
        seg_results.append({
            "segment": seg,
            "n_docs": len(sub),
            "churn_rate": round(sub["is_churn"].mean(), 4),
            "competitor_rate": round(sub["is_competitor"].mean(), 4),
            "medical_rate": round(sub["is_medical_frame"].mean(), 4),
            "ingredient_rate": round(sub["is_ingredient_frame"].mean(), 4),
            "base_prob": round(base_prob, 4),
            "switching_probability": round(switching_prob, 4)
        })

    return pd.DataFrame(seg_results), trend_multiplier


if __name__ == "__main__":
    voc_df, trend_df = load_features()
    print(f"VoC features: {len(voc_df)} | Trend features: {len(trend_df)}")

    X, y, feature_cols = build_model_features(voc_df)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Segment classes: {list(le.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # find optimal regularization
    print("\nRegularization Grid Search:")
    clf, best_C = find_optimal_regularization(X_scaled, y_encoded)

    # feature coefficients with optimal C
    print("\nFeature Coefficients per Segment (optimal C):")
    coef_df = pd.DataFrame(
        clf.coef_,
        columns=feature_cols,
        index=le.classes_
    )
    print(coef_df.round(3).to_string())

    # find optimal trend weights
    print("\nTrend Weight Grid Search:")
    best_w1, best_w2 = find_optimal_trend_weights(
        voc_df, trend_df, clf, scaler, le, feature_cols
    )

    # compute final switching probability
    print("\nFinal Switching Probability (optimal C + optimal weights):")
    seg_prob_df, trend_mult = compute_final_switching_probability(
        voc_df, trend_df, clf, scaler, le, feature_cols, best_w1, best_w2
    )
    print(seg_prob_df.to_string(index=False))

    seg_prob_df.to_csv(f"{OUTPUT_DIR}/switching_prob_regression.csv",
                       index=False, encoding="utf-8-sig")
    print(f"\n {OUTPUT_DIR}/switching_prob_regression.csv")