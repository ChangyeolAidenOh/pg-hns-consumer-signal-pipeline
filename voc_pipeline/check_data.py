import pandas as pd

blog = pd.read_csv("data/raw/naver_blog_hns.csv")
cafe = pd.read_csv("data/raw/naver_cafe_hns.csv")
yt = pd.read_csv("data/raw/youtube_comments_hns.csv")

print(blog[["title", "description"]].head(10).to_string())
print()
print(yt[["comment"]].head(20).to_string())