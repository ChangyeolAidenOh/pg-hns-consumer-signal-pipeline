import os
import time
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

OUTPUT_DIR = "data/raw"


def search_videos(query, max_results=20):
    """
    Search YouTube videos related to a query.

    Args:
        query: search keyword
        max_results: maximum number of videos to return

    Returns:
        list of video metadata dicts
    """
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        order="relevance",
        relevanceLanguage="ko",
        regionCode="KR"
    )
    response = request.execute()
    videos = []
    for item in response.get("items", []):
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"],
            "query": query
        })
    return videos


def get_comments(video_id, max_results=100):
    """
    Collect top-level comments from a YouTube video.
    Args:
        video_id: YouTube video ID
        max_results: maximum number of comments to collect
    Returns:
        list of comment dicts
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            order="relevance",
            textFormat="plainText"
        )
        response = request.execute()
        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment": top["textDisplay"],
                "author": top["authorDisplayName"],
                "likes": top["likeCount"],
                "published_at": top["publishedAt"],
            })
        # paginate through remaining comments
        while "nextPageToken" in response and len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
                order="relevance",
                textFormat="plainText"
            )
            response = request.execute()
            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment": top["textDisplay"],
                    "author": top["authorDisplayName"],
                    "likes": top["likeCount"],
                    "published_at": top["publishedAt"],
                })
    except Exception as e:
        print(f"Comment collection error (video_id: {video_id}): {e}")
    return comments


def collect_youtube_data(queries, max_videos=10, max_comments=100):
    """
    Collect videos and comments for a list of queries.
    Args:
        queries: list of search keywords
        max_videos: max videos per query
        max_comments: max comments per video
    Returns:
        tuple of (videos_df, comments_df)
    """
    all_videos = []
    all_comments = []

    for query in queries:
        print(f"Searching videos: {query}")
        videos = search_videos(query, max_results=max_videos)
        all_videos.extend(videos)

        for video in videos:
            vid = video["video_id"]
            print(f"  Collecting comments: {video['title'][:40]}...")
            comments = get_comments(vid, max_results=max_comments)
            for c in comments:
                c["query"] = query
                c["video_title"] = video["title"]
                c["channel"] = video["channel"]
            all_comments.extend(comments)
            time.sleep(0.3)  # rate limiting between requests

    return pd.DataFrame(all_videos), pd.DataFrame(all_comments)


if __name__ == "__main__":
    queries = [
        "헤드앤숄더 리뷰",
        "비듬샴푸 추천",
        "안티트로 vs 헤드앤숄더",
        "두피케어 샴푸 비교",
    ]

    videos_df, comments_df = collect_youtube_data(
        queries,
        max_videos=10,
        max_comments=100
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    videos_df.to_csv(f"{OUTPUT_DIR}/youtube_videos_hns.csv", index=False, encoding="utf-8-sig")
    comments_df.to_csv(f"{OUTPUT_DIR}/youtube_comments_hns.csv", index=False, encoding="utf-8-sig")

    print(f"  Videos: {len(videos_df)} records")
    print(f"  Comments: {len(comments_df)} records")
    print(f"{OUTPUT_DIR}/youtube_videos_hns.csv")
    print(f"{OUTPUT_DIR}/youtube_comments_hns.csv")