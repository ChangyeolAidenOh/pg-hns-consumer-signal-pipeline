import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sktime.classification.kernel_based import RocketClassifier
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "trend_pipeline/data/processed"


def load_features():
    df = pd.read_csv(f"{PROCESSED_DIR}/trend_features.csv",
                     index_col=0, parse_dates=True)
    df = df.fillna(0)
    print(f"Loaded: {df.shape} | {df.index.min()} ~ {df.index.max()}")
    return df


def make_risk_labels(df):
    """
    Generate binary risk labels based on 안티트로 / 헤드앤숄더 ratio.
    Risk = 1 if 안티트로샴푸 > 헤드앤숄더샴푸 * 0.5
    """
    labels = (df["안티트로샴푸"] > df["헤드앤숄더샴푸"] * 0.5).astype(int)
    print(f"Risk months: {labels.sum()} / {len(labels)}")
    return labels


def run_chronos_forecast(df, target_col="헤드앤숄더샴푸", forecast_horizon=12):
    """
    Part A: Zero-shot time series forecasting using Amazon Chronos.
    Forecasts next `forecast_horizon` months of target column.
    """
    print("\n Chronos Forecast")

    try:
        from chronos import BaseChronosPipeline
    except ImportError:
        print(" Skipping Part A")
        return None

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32
    )

    series = torch.tensor(df[target_col].values, dtype=torch.float32)
    forecast = pipeline.predict(series.unsqueeze(0), forecast_horizon)

    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=forecast_horizon + 1,
                                 freq="ME")[1:]

    results = pd.DataFrame({
        "date": future_dates,
        "forecast_median": median,
        "forecast_low": low,
        "forecast_high": high
    })

    print(f"  Forecast horizon: {forecast_horizon} months")
    print(results.to_string(index=False))

    results.to_csv(f"{PROCESSED_DIR}/chronos_forecast.csv",
                   index=False, encoding="utf-8-sig")
    return results


def run_trend_analysis(df):
    """
    Part B: Structural trend analysis of HNS and competitor dynamics.
    Focuses on business implications rather than ML classification,
    given the fundamental data constraint (n=76, risk labels concentrated in tail).
    Five analytical dimensions:
      1. Antitro reversal timing and acceleration speed
      2. HNS product line lifecycle (rise and decline by SKU)
      3. Competitor brand trajectory vs 2020 baseline
      4. Shampoo category structural decline
      5. Symptom-category keyword language shift
    """
    print("\n Structural Trend Analysis")

    results = {}

    # 1. Antitro reversal: first month when 안티트로 exceeded 헤드앤숄더샴푸
    df["antitro_ratio"] = df["안티트로샴푸"] / df["헤드앤숄더샴푸"].replace(0, 1)
    first_entry = df[df["안티트로샴푸"] > 0].index
    first_reversal = df[df["antitro_ratio"] >= 1.0].index

    if len(first_reversal) > 0:
        months_to_reversal = (first_reversal[0] - first_entry[0]).days // 30
        print(f"\n  [1] Antitro first reversal: {first_reversal[0].strftime('%Y-%m')}")
        print(f"      Current ratio (antitro/hns): {df['antitro_ratio'].iloc[-1]:.3f}")
        print(f"      Months from first entry to reversal: {months_to_reversal} months")
    results["first_reversal"] = first_reversal[0].strftime("%Y-%m") if len(first_reversal) > 0 else "N/A"
    results["current_ratio"] = round(df["antitro_ratio"].iloc[-1], 3)
    results["months_to_reversal"] = months_to_reversal if len(first_reversal) > 0 else None


    # 2. HNS product line lifecycle: annual average by SKU
    # Tracks which lines are growing, plateauing, or declining
    print("\n  [2] HNS product line annual average by SKU")
    hns_lines = ["헤드앤숄더샴푸", "헤드앤숄더클리니컬스트렝스",
                 "헤드앤숄더프로페셔널", "헤드앤숄더차콜"]
    line_labels = ["Core", "Clinical", "Professional", "Charcoal"]
    for year in [2021, 2023, 2025, 2026]:
        sub = df[df.index.year == year]
        if len(sub) == 0:
            continue
        vals = {label: round(sub[col].mean(), 1)
                for col, label in zip(hns_lines, line_labels)}
        print(f"      {year}: " + " | ".join([f"{k}={v}" for k, v in vals.items()]))

    # 3. Competitor brand trajectory: % change from 2020 baseline
    # Identifies which brands are gaining or losing share in the same category
    print("\n  [3] Competitor brand change vs 2020 baseline")
    competitors = ["닥터그루트", "케라시스", "팬틴"]
    base_2020 = df[df.index.year == 2020][competitors].mean()
    latest = df[df.index.year == 2026][competitors].mean()
    for col in competitors:
        if base_2020[col] > 0:
            chg = (latest[col] - base_2020[col]) / base_2020[col] * 100
            print(f"      {col}: {base_2020[col]:.1f} → {latest[col]:.1f} ({chg:+.1f}%)")

    # 4. Shampoo category structural decline: annual category click volume
    # Measures whether the overall shampoo market is shrinking
    print("\n  [4] Shampoo category click volume by year")
    base_cat = df[df.index.year == 2020]["category_click"].mean()
    for year in range(2020, 2027):
        sub = df[df.index.year == year]
        if len(sub) == 0:
            continue
        avg = sub["category_click"].mean()
        chg = (avg - base_cat) / base_cat * 100
        print(f"      {year}: {avg:.1f} ({chg:+.1f}%)")


    # 5. Symptom-category keyword language shift

    # Tracks how consumers describe their scalp problems over time
    # Rise of antitro-specific language signals category redefinition

    print("\n  [5] Symptom-category keyword language shift vs 2020")
    symptom_cols = ["비듬샴푸", "지루성두피샴푸", "안티트로샴푸"]
    symptom_labels = ["Dandruff Shampoo", "Seborrheic Shampoo", "Antitro Shampoo"]
    base_2020_s = df[df.index.year == 2020][symptom_cols].mean()
    latest_s = df[df.index.year == 2026][symptom_cols].mean()
    for col, label in zip(symptom_cols, symptom_labels):
        if base_2020_s[col] > 0:
            chg = (latest_s[col] - base_2020_s[col]) / base_2020_s[col] * 100
            print(f"      {label}: {base_2020_s[col]:.1f} => {latest_s[col]:.1f} ({chg:+.1f}%)")
        else:
            print(f"      {label}: 0.0 => {latest_s[col]:.1f} (new entrant)")

    # save summary for Layer 3 input
    summary = pd.DataFrame([results])
    summary.to_csv(f"{PROCESSED_DIR}/trend_analysis_summary.csv",
                   index=False, encoding="utf-8-sig")
    return results


if __name__ == "__main__":
    df = load_features()
    labels = make_risk_labels(df)

    # Part A: Chronos zero-shot forecast
    forecast_results = run_chronos_forecast(df, target_col="헤드앤숄더샴푸",
                                            forecast_horizon=12)

    # Part B: Structural trend analysis
    trend_results = run_trend_analysis(df)

    print("\n chronos_forecast.csv, trend_analysis_summary.csv")