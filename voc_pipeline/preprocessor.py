import re
import ast
import pandas as pd
from kiwipiepy import Kiwi

kiwi = Kiwi()
kiwi.add_user_word("안티트로", "NNP", 0)

# Set stop words tailored for the Head & Shoulders (hair care/scalp) domain
STOPWORDS = {
    # 1. Category and brand noise (words that appear in almost all documents)
    "헤드앤숄더", "헤숄", "샴푸", "제품", "브랜드", "상품",

    # 2. General review/shopping/daily terms (meaningless nouns for insight derivation)
    "사용", "후기", "추천", "구매", "내돈내산", "생각", "정도", "요즘", "오늘", "최근",
    "사람", "부분", "이름", "이유", "마음", "사실", "기존", "이것", "저것", "그것",

    # 3. General beauty/hair noise
    "머리", "헤어", "뷰티", "화장품",

    # 4. Ads/spam/shopping platform noise (to prevent data contamination)
    "최저가", "최저", "링크", "프로필", "프로필링크", "혜택", "진행중", "할인", "세일",
    "이벤트", "광고", "협찬", "올리브영", "다이소", "쿠팡", "코스트코", "약국",

    # 5. Platform-specific noise (YouTube, blogs, etc.)
    "구독", "영상", "좋아요", "알림", "시청", "채널", "방송", "유튜브", "shorts",
    "포스팅", "이웃", "목차", "안녕", "안녕하세요",

    # 6. Exclamations and adverbs (fallback for Kiwi's noun extraction)
    "진짜", "정말", "완전", "대박", "레전드", "꿀팁", "ㅠㅠ", "ㅋㅋ", "ㅎㅎ",
    
    # 7. Context filler nouns
    "때문", "경우", "통해", "위해", "관련", "개입",

    # 8. Remaining shopping platform noise (confirmed from LDA output)
    "옵션", "평점", "적립", "가격", "최대", "보라색", "두피",

}


# Text cleaning step
def clean_text(text):
    """
    Remove HTML tags, URLs, and special characters from raw text.
    Args:
        text: raw input string
    Returns:
        cleaned string
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Morphological analysis
def extract_tokens(text):
    """
    Extract nouns and adjective-noun pairs using kiwipiepy.
    Args:
        text: cleaned Korean text
    Returns:
        dict with keys:
            'nouns': list of noun tokens
            'adj_noun_pairs': list of (adjective, noun) tuples
    """
    if not text or len(text) < 2:
        return {"nouns": [], "adj_noun_pairs": []}
    try:
        result = kiwi.analyze(text)
        tokens = result[0][0]
        nouns, pairs = [], []
        for i, token in enumerate(tokens):
            # Extract nouns: minimum 2 characters, exclude stopwords
            if token.tag in ("NNG", "NNP") and len(token.form) >= 2:
                if token.form not in STOPWORDS:
                    nouns.append(token.form)
            # Extract adjective-noun pairs for sentiment-bearing phrases
            if token.tag in ("VA", "XR") and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.tag in ("NNG", "NNP") and len(next_token.form) >= 2:
                    if next_token.form not in STOPWORDS:
                        pairs.append((token.form, next_token.form))
        return {"nouns": nouns, "adj_noun_pairs": pairs}
    except:
        return {"nouns": [], "adj_noun_pairs": []}


# Preprocessing step
def make_unigram(nouns):
    """Single noun tokens filtered by stopwords."""
    return [n for n in nouns if n not in STOPWORDS]


def make_bigram(nouns):
    """Consecutive noun pairs (e.g., 차앤박_앰플)."""
    bigrams = []
    for i in range(len(nouns) - 1):
        bigrams.append(f"{nouns[i]}_{nouns[i + 1]}")
    return bigrams


def make_unibi_mix(nouns):
    """Union of unigram and bigram tokens."""
    return make_unigram(nouns) + make_bigram(nouns)


def make_adj_noun(pairs):
    """Adjective-noun compound phrases (e.g., 순한_클렌저)."""
    return [f"{adj}_{noun}" for adj, noun in pairs]


# Preprocessing per source
def process_row(row, text_cols):
    """
    Process a single row by combining specified text columns
    and applying all preprocessing modes.
    Args:
        row: DataFrame row
        text_cols: list of column names to combine as input text
    Returns:
        dict with raw_text and four token mode lists
    """
    combined = " ".join([str(row.get(col, "")) for col in text_cols])
    cleaned = clean_text(combined)
    tokens = extract_tokens(cleaned)
    nouns = tokens["nouns"]
    pairs = tokens["adj_noun_pairs"]
    return {
        "raw_text": cleaned,
        "unigram": nouns,
        "bigram": make_bigram(nouns),
        "unibi_mix": make_unibi_mix(nouns),
        "adj_noun": make_adj_noun(pairs),
    }


def preprocess_naver(df, source_name):
    """
    Preprocess Naver blog or cafe posts.
    Combines title and description columns as input text.
    """
    records = []
    for _, row in df.iterrows():
        processed = process_row(row, ["title", "description"])
        records.append({
            "source": source_name,
            "date": str(row.get("postdate", "")),
            "query": row.get("query", ""),
            **processed
        })
    return pd.DataFrame(records)


def preprocess_youtube(df):
    """
    Preprocess YouTube comments.
    Uses comment text as input.
    """
    records = []
    for _, row in df.iterrows():
        processed = process_row(row, ["comment"])
        records.append({
            "source": "youtube",
            "date": str(row.get("published_at", ""))[:10],
            "query": row.get("query", ""),
            "video_title": row.get("video_title", ""),
            "likes": row.get("likes", 0),
            **processed
        })
    return pd.DataFrame(records)


# Main execution workflow
if __name__ == "__main__":
    RAW_DIR = "voc_pipeline/data/raw"
    PROCESSED_DIR = "voc_pipeline/data/processed"

    # Load and process Naver Blog data
    blog_df = pd.read_csv(f"{RAW_DIR}/naver_blog_hns.csv")
    blog_processed = preprocess_naver(blog_df, "blog")

    # Load and process Naver Cafe data
    cafe_df = pd.read_csv(f"{RAW_DIR}/naver_cafe_hns.csv")
    cafe_processed = preprocess_naver(cafe_df, "cafearticle")

    # Load and process YouTube comments data
    yt_df = pd.read_csv(f"{RAW_DIR}/youtube_comments_hns.csv")
    yt_processed = preprocess_youtube(yt_df)

    # Concatenate all processed DataFrames
    all_df = pd.concat([blog_processed, cafe_processed, yt_processed], ignore_index=True)

    # Save processed output
    all_df.to_csv(f"{PROCESSED_DIR}/hns_processed.csv", index=False, encoding="utf-8-sig")
    print(f"Data successfully saved to: {PROCESSED_DIR}/hns_processed.csv")

    # Sample output for verification (filter rows with > 3 unigrams)
    sample = all_df[
        all_df["unigram"].apply(
            lambda x: len(ast.literal_eval(str(x)) if isinstance(x, str) else x) > 3
        )
    ].iloc[0]

    print("\n# Sample token extraction #")
    print(f"Raw text: {sample['raw_text'][:60]}")
    for mode in ["unigram", "bigram", "unibi_mix", "adj_noun"]:
        val = sample[mode]
        if isinstance(val, str):
            val = ast.literal_eval(val)
        print(f"[{mode}]: {val[:6]}")