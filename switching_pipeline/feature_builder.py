import ast
import pandas as pd
import numpy as np

PROCESSED_DIR_VOC = "voc_pipeline/data/processed"
PROCESSED_DIR_TREND = "trend_pipeline/data/processed"
OUTPUT_DIR = "switching_pipeline/data"


def parse_list(val):
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []


def build_voc_features(causal_df):
    """
    Build document-level features from Layer 1 VoC signals.
    Each document gets a feature vector encoding:
    - signal type (churn risk / positive / neutral)
    - source channel
    - competitor mention
    - key keyword flags (medical frame, ingredient frame, comparison frame)
    """
    records = []
    for _, row in causal_df.iterrows():
        text = str(row.get("raw_text", ""))
        unigram = parse_list(row.get("unigram", "[]"))
        unigram_set = set(unigram)

        # channel encoding
        source = row.get("source", "")
        is_youtube = int(source == "youtube")
        is_blog = int(source == "blog")
        is_cafe = int(source == "cafearticle")

        # signal encoding
        signal = row.get("signal_type", "중립")
        is_churn = int(signal == "이탈위험")
        is_positive = int(signal == "긍정")

        # frame detection
        # medical frame: consumer treats scalp issue as clinical condition
        medical_keywords = {"지루성두피염", "피부과", "치료", "항진균", "질환", "처방", "병원"}
        is_medical_frame = int(bool(medical_keywords & unigram_set) or
                               any(kw in text for kw in medical_keywords))

        # ingredient frame: consumer scrutinizes product ingredients
        ingredient_keywords = {"설페이트", "계면활성제", "약산성", "성분", "소듐", "화학"}
        is_ingredient_frame = int(bool(ingredient_keywords & unigram_set) or
                                  any(kw in text for kw in ingredient_keywords))

        # comparison frame: consumer actively compares brands
        comparison_keywords = {"안티트로", "니조랄", "대신", "갈아탔", "갈아탈", "비교"}
        is_comparison_frame = int(bool(comparison_keywords & unigram_set) or
                                  any(kw in text for kw in comparison_keywords))

        # competitor mention
        is_competitor = int(row.get("competitor_mentioned", False))

        # churn score
        churn_score = int(row.get("churn_score", 0))
        positive_score = int(row.get("positive_score", 0))

        records.append({
            "source": source,
            "date": row.get("date", ""),
            "raw_text": text[:100],
            "is_youtube": is_youtube,
            "is_blog": is_blog,
            "is_cafe": is_cafe,
            "is_churn": is_churn,
            "is_positive": is_positive,
            "is_medical_frame": is_medical_frame,
            "is_ingredient_frame": is_ingredient_frame,
            "is_comparison_frame": is_comparison_frame,
            "is_competitor": is_competitor,
            "churn_score": churn_score,
            "positive_score": positive_score,
        })

    return pd.DataFrame(records)


def build_trend_features():
    """
    Build monthly trend features from Layer 2.
    Returns key signals: antitro ratio, hns decline rate,
    category trend, Chronos forecast.
    """
    trend = pd.read_csv(f"{PROCESSED_DIR_TREND}/trend_features.csv",
                        index_col=0, parse_dates=True)
    trend = trend.fillna(0)

    # antitro pressure: ratio of antitro to hns core
    trend["antitro_ratio"] = trend["안티트로샴푸"] / trend["헤드앤숄더샴푸"].replace(0, 1)

    # hns momentum: 3-month rolling change rate
    trend["hns_momentum"] = trend["헤드앤숄더샴푸"].pct_change(3)

    # category health: 3-month rolling change rate
    trend["category_momentum"] = trend["category_click"].pct_change(3)

    # competitor pressure: combined competitor search volume
    trend["competitor_pressure"] = trend["닥터그루트"] + trend["케라시스"] + trend["팬틴"]

    # medical category growth: 지루성두피샴푸 momentum
    trend["medical_momentum"] = trend["지루성두피샴푸"].pct_change(3)

    cols = ["antitro_ratio", "hns_momentum", "category_momentum",
            "competitor_pressure", "medical_momentum",
            "헤드앤숄더샴푸", "안티트로샴푸", "지루성두피샴푸"]

    return trend[cols].dropna()


def assign_segment(row):
    """
    Assign consumer segment based on VoC feature combination.
    Segments:
    - Active Switcher:  comparison frame + competitor mention + churn signal
    - At-risk:          medical/ingredient frame + churn signal (no direct comparison yet)
    - Passive User:     neutral signal, no strong frame
    - Loyal:            positive signal + no churn indicators
    """
    if row["is_comparison_frame"] and row["is_competitor"] and row["is_churn"]:
        return "Active Switcher"
    elif (row["is_medical_frame"] or row["is_ingredient_frame"]) and row["is_churn"]:
        return "At-risk"
    elif row["is_positive"] and not row["is_churn"]:
        return "Loyal"
    else:
        return "Passive User"


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load VoC causal signals
    causal_df = pd.read_csv(f"{PROCESSED_DIR_VOC}/hns_causal_signals.csv")
    print(f"VoC documents: {len(causal_df)}")

    # build VoC features
    voc_features = build_voc_features(causal_df)
    voc_features["segment"] = voc_features.apply(assign_segment, axis=1)

    print("\nSegment Distribution:")
    seg_counts = voc_features["segment"].value_counts()
    for seg, cnt in seg_counts.items():
        pct = cnt / len(voc_features) * 100
        print(f"  {seg}: {cnt} ({pct:.1f}%)")

    print("\nSegment × Source:")
    print(voc_features.groupby(["segment", "source"]).size().unstack(fill_value=0).to_string())

    # build trend features
    trend_features = build_trend_features()
    print(f"\nTrend features: {trend_features.shape}")
    print(f"Latest antitro_ratio: {trend_features['antitro_ratio'].iloc[-1]:.3f}")
    print(f"Latest hns_momentum: {trend_features['hns_momentum'].iloc[-1]:.3f}")

    # save
    voc_features.to_csv(f"{OUTPUT_DIR}/voc_features.csv",
                        index=False, encoding="utf-8-sig")
    trend_features.to_csv(f"{OUTPUT_DIR}/trend_features_processed.csv",
                          encoding="utf-8-sig")

    print(f"\n {OUTPUT_DIR}/voc_features.csv")
    print(f" {OUTPUT_DIR}/trend_features_processed.csv")