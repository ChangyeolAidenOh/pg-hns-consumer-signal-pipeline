import os
import unicodedata
import pandas as pd


RAW_DIR = "trend_pipeline/data/raw"


def normalize(f):
    return unicodedata.normalize("NFC", f)


def load_category_trend():
    """
    Load Naver DataLab category click trend (샴푸 분야통계).
    Returns monthly time series DataFrame.
    """
    files = sorted([
        f for f in os.listdir(RAW_DIR)
        if normalize(f).startswith("분야통계_샴푸") and f.endswith(".csv")
    ])
    dfs = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        df = pd.read_csv(path, encoding="utf-8-sig", skiprows=8,
                         names=["date", "click"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["click"] = pd.to_numeric(df["click"], errors="coerce")
        dfs.append(df)
    df_all = pd.concat(dfs).drop_duplicates("date").sort_values("date")
    df_all = df_all.set_index("date").resample("ME").mean()
    df_all.columns = ["category_click"]
    return df_all


def load_brand_trend():
    """
    Load Naver DataLab brand search trend (헤드앤숄더 검색어통계).
    Returns monthly time series DataFrame with 5 keyword columns.
    """
    files = sorted([
        f for f in os.listdir(RAW_DIR)
        if normalize(f).startswith("naver_shopping_keyword_헤드앤숄더")
        and f.endswith(".csv")
    ])
    dfs = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        df = pd.read_csv(path, encoding="utf-8-sig", skiprows=8)
        df.columns = ["date", "헤드앤숄더샴푸", "헤드앤숄더차콜",
                      "헤드앤숄더", "헤드앤숄더프로페셔널", "헤드앤숄더클리니컬스트렝스"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        dfs.append(df)
    df_all = pd.concat(dfs).drop_duplicates("date").sort_values("date")
    df_all = df_all.set_index("date").resample("ME").mean()
    return df_all


def load_symptom_trend():
    """
    Load Naver DataLab symptom category search trend.
    Returns monthly time series DataFrame with 5 keyword columns.
    """
    files = sorted([
        f for f in os.listdir(RAW_DIR)
        if normalize(f).startswith("naver_shopping_keyword_증상카테고리")
        and f.endswith(".csv")
    ])
    dfs = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        df = pd.read_csv(path, encoding="utf-8-sig", skiprows=8)
        df.columns = ["date", "비듬샴푸", "지루성두피샴푸",
                      "안티트로샴푸", "지성두피샴푸", "비듬"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        dfs.append(df)
    df_all = pd.concat(dfs).drop_duplicates("date").sort_values("date")
    df_all = df_all.set_index("date").resample("ME").mean()
    return df_all


def load_competition_trend():
    """
    Load Naver search trend brand competition data.
    Returns monthly time series DataFrame.
    """
    import unicodedata as ud
    files = [
        f for f in os.listdir(RAW_DIR)
        if ud.normalize("NFC", f) == "네이버_검색어트렌드_브랜드경쟁구도.xlsx"
    ]
    if not files:
        return pd.DataFrame()
    path = os.path.join(RAW_DIR, files[0])
    df_raw = pd.read_excel(path, sheet_name="개요", header=None)
    data = df_raw.iloc[7:].copy()
    df = pd.DataFrame({
        "date": pd.to_datetime(data[0], errors="coerce"),
        "팬틴": pd.to_numeric(data[1], errors="coerce"),
        "헤드앤숄더_경쟁": pd.to_numeric(data[3], errors="coerce"),
        "닥터그루트": pd.to_numeric(data[5], errors="coerce"),
        "케라시스": pd.to_numeric(data[7], errors="coerce"),
        "두피케어카테고리": pd.to_numeric(data[9], errors="coerce"),
    })
    df = df.dropna(subset=["date"]).set_index("date").resample("ME").mean()
    return df


def load_all():
    """
    Merge all trend data sources into a single monthly feature table.
    Returns unified DataFrame indexed by month.
    """
    category = load_category_trend()
    brand    = load_brand_trend()
    symptom  = load_symptom_trend()
    competition = load_competition_trend()

    df = category.join(brand, how="outer")
    df = df.join(symptom, how="outer")
    if not competition.empty:
        df = df.join(competition, how="outer")

    df = df.sort_index()
    df = df[df.index >= "2020-01-01"]
    return df


if __name__ == "__main__":
    df = load_all()
    print(f"Trend feature table: {df.shape}")
    print(df.head(10).to_string())
    df.to_csv("trend_pipeline/data/processed/trend_features.csv",
              encoding="utf-8-sig")
    print("Saved: trend_pipeline/data/processed/trend_features.csv")