import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

OUTPUT_DIR = "data/raw"


def search_naver(query, source="blog", display=100, start=1):
    """
    Query Naver Search API for blog or cafe posts.
    Args:
        query: search keyword
        source: 'blog' or 'cafearticle'
        display: number of results per request (max 100)
        start: start index for pagination
    """
    url = f"https://openapi.naver.com/v1/search/{source}"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": "date"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def collect_naver_data(queries, source="blog"):
    """
    Collect posts from Naver blog or cafe for a list of queries.
    Args:
        queries: list of search keywords
        source: 'blog' or 'cafearticle'
    Returns:
        DataFrame of collected posts
    """
    # Keywords filtering
    RELEVANCE_KEYWORDS = [
        '헤드앤숄더', '비듬', '두피각질', '지루성두피',
        '안티트로', '차콜샴푸', '비듬샴푸', '두피샴푸', '각질샴푸'
    ]

    def is_relevant(item):
        text = (item.get("title", "") + item.get("description", "")).lower()
        return any(kw in text for kw in RELEVANCE_KEYWORDS)

    all_items = []
    for query in queries:
        print(f"Collecting [{source}]: {query}")
        result = search_naver(query, source=source, display=100)
        if result and "items" in result:
            for item in result["items"]:
                if is_relevant(item):
                    item["query"] = query
                    item["source"] = source
                    all_items.append(item)
        time.sleep(0.5)  # rate limiting between API calls
    return pd.DataFrame(all_items)


if __name__ == "__main__":
    # head&shoulders-related search queries
    hns_queries = [
        # Product-specific queries
        "헤드앤숄더 후기",
        "헤드앤숄더 샴푸 사용후기",
        "헤드앤숄더 차콜 후기",
        "헤드앤숄더 바꿨어요",
        "헤드앤숄더 대신",

        # Positional queries
        "비듬샴푸 추천",
        "비듬 없애는 샴푸",
        "두피각질 샴푸",

        # Competitive comparison queries
        "안티트로 샴푸 후기",
        "지루성두피 샴푸 추천",
    ]

    # Collect blog posts
    blog_df = collect_naver_data(hns_queries, source="blog")
    print(f"blog {len(blog_df)} records")

    # Collect cafe posts
    cafe_df = collect_naver_data(hns_queries, source="cafearticle")
    print(f"cafe {len(cafe_df)} records")

    # Save to raw data directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    blog_df.to_csv(f"{OUTPUT_DIR}/naver_blog_hns.csv", index=False, encoding="utf-8-sig")
    cafe_df.to_csv(f"{OUTPUT_DIR}/naver_cafe_hns.csv", index=False, encoding="utf-8-sig")
    print(f"{OUTPUT_DIR}/naver_blog_hns.csv")
    print(f"{OUTPUT_DIR}/naver_cafe_hns.csv")