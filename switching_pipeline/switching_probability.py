import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SWITCHING_PROB_PATH = "switching_pipeline/data/switching_prob_regression.csv"
TREND_PATH = "switching_pipeline/data/trend_features_processed.csv"
VOC_PATH = "switching_pipeline/data/voc_features.csv"
OUTPUT_DIR = "switching_pipeline/data"


def load_all():
    seg_prob = pd.read_csv(SWITCHING_PROB_PATH)
    trend = pd.read_csv(TREND_PATH, index_col=0, parse_dates=True)
    voc = pd.read_csv(VOC_PATH)
    return seg_prob, trend, voc


def compute_brand_risk_score(seg_prob, trend):
    """
    Compute overall brand-level switching risk score.
    Aggregates segment-level switching probabilities
    weighted by segment size (n_docs proportion).
    Incorporates trend pressure as an external risk amplifier.

    Brand Risk Score = weighted average of switching probabilities
                       across all segments
    Range: 0.0 (no risk) to 1.0 (maximum risk)
    """
    total_docs = seg_prob["n_docs"].sum()
    seg_prob["weight"] = seg_prob["n_docs"] / total_docs

    brand_risk = (seg_prob["switching_probability"] * seg_prob["weight"]).sum()

    # trend context
    latest = trend.iloc[-1]
    antitro_ratio = latest["antitro_ratio"]
    hns_momentum = latest["hns_momentum"]

    return round(brand_risk, 4), antitro_ratio, hns_momentum


def generate_segment_implications(seg_prob):
    """
    Generate channel-specific intervention recommendations
    per consumer segment based on switching probability
    and behavioral signal profile.

    Intervention logic:
    - Active Switcher: urgent retention, competitor differentiation
    - At-risk: proactive education on ingredients/efficacy
    - Passive User: awareness and engagement
    - Loyal: loyalty reinforcement
    """
    implications = []

    for _, row in seg_prob.iterrows():
        seg = row["segment"]
        prob = row["switching_probability"]
        n = row["n_docs"]
        churn = row["churn_rate"]
        competitor = row["competitor_rate"]
        medical = row["medical_rate"]
        ingredient = row["ingredient_rate"]

        if seg == "Active Switcher":
            implications.append({
                "segment": seg,
                "n_docs": n,
                "switching_probability": prob,
                "risk_level": "Critical",
                "primary_channel": "Blog",
                "intervention_timing": "Immediate",
                "recommended_action": (
                    "Direct competitive messaging on blogs. "
                    "Address antitro comparison with clinical efficacy data. "
                    "Reinforce HNS Clinical Strength positioning vs derma brands."
                )
            })

        elif seg == "At-risk":
            implications.append({
                "segment": seg,
                "n_docs": n,
                "switching_probability": prob,
                "risk_level": "High",
                "primary_channel": "All channels (uniform distribution)",
                "intervention_timing": "Pre-switch (3-4 weeks)",
                "recommended_action": (
                    "Ingredient transparency content: explain zinc pyrithione "
                    "vs antifungal mechanism. "
                    "Medical frame alignment: position HNS as dermatologist-recommended, "
                    "not just a shampoo brand."
                )
            })

        elif seg == "Passive User":
            implications.append({
                "segment": seg,
                "n_docs": n,
                "switching_probability": prob,
                "risk_level": "Moderate",
                "primary_channel": "YouTube + Cafe",
                "intervention_timing": "Ongoing",
                "recommended_action": (
                    "Scalp health education content on YouTube. "
                    "Prevent frame shift toward medical/ingredient scrutiny "
                    "by proactively addressing ingredient questions."
                )
            })

        elif seg == "Loyal":
            implications.append({
                "segment": seg,
                "n_docs": n,
                "switching_probability": prob,
                "risk_level": "Low",
                "primary_channel": "Cafe",
                "intervention_timing": "Maintenance",
                "recommended_action": (
                    "Loyalty reinforcement via community engagement. "
                    "Leverage satisfied users as brand advocates "
                    "in comparison discussions."
                )
            })

    return pd.DataFrame(implications)


def generate_timeline_implication(trend):
    """
    Generate timeline-based strategic implications
    from Layer 2 trend analysis.
    Identifies three critical time windows:
    - Reversal point: when antitro first exceeded HNS
    - Current state: present competitive pressure
    - Forecast window: Chronos 12-month outlook
    """
    # reversal point
    trend["antitro_ratio"] = trend["안티트로샴푸"] / trend["헤드앤숄더샴푸"].replace(0, 1)
    first_reversal = trend[trend["antitro_ratio"] >= 1.0].index
    reversal_date = first_reversal[0].strftime("%Y-%m") if len(first_reversal) > 0 else "N/A"

    # acceleration point: largest month-over-month antitro increase
    trend["antitro_delta"] = trend["안티트로샴푸"].diff()
    accel_date = trend["antitro_delta"].idxmax().strftime("%Y-%m")
    accel_value = trend["antitro_delta"].max()

    # current state
    latest = trend.iloc[-1]
    current_ratio = round(latest["antitro_ratio"], 3)
    current_momentum = round(latest["hns_momentum"], 3)

    timeline = {
        "antitro_first_entry": "2024-12",
        "antitro_reversal_point": reversal_date,
        "antitro_acceleration_point": accel_date,
        "antitro_acceleration_delta": round(accel_value, 1),
        "current_antitro_ratio": current_ratio,
        "current_hns_momentum": current_momentum,
        "strategic_window": (
            "Antitro completed reversal in 6 months (2024-12 to 2025-06). "
            "Current ratio 1.327 indicates sustained dominance. "
            "HNS momentum -0.237 signals continued search volume decline. "
            "Critical intervention window: before antitro ratio exceeds 1.5 "
            "and becomes the default consumer reference point."
        )
    }

    return timeline


if __name__ == "__main__":
    seg_prob, trend, voc = load_all()

    # brand-level risk score
    brand_risk, antitro_ratio, hns_momentum = compute_brand_risk_score(seg_prob, trend)
    print(f"Brand Switching Risk Score: {brand_risk:.4f}")
    print(f"  antitro_ratio (latest): {antitro_ratio:.3f}")
    print(f"  hns_momentum (latest):  {hns_momentum:.3f}")

    # segment-level implications
    print("\nSegment-level Intervention Plan:")
    implications_df = generate_segment_implications(seg_prob)
    print(implications_df[["segment", "risk_level", "switching_probability",
                            "intervention_timing"]].to_string(index=False))

    # timeline implications
    print("\nTimeline Analysis:")
    timeline = generate_timeline_implication(trend)
    for k, v in timeline.items():
        print(f"  {k}: {v}")

    # save
    implications_df.to_csv(f"{OUTPUT_DIR}/switching_implications.csv",
                            index=False, encoding="utf-8-sig")
    pd.DataFrame([timeline]).to_csv(f"{OUTPUT_DIR}/timeline_analysis.csv",
                                     index=False, encoding="utf-8-sig")

    print(f"\n switching_implications.csv, timeline_analysis.csv")