import os
import requests
from dotenv import load_dotenv
from naver_news import NaverNewsClient

load_dotenv()

client = NaverNewsClient(
    client_id=os.getenv("NAVER_CLIENT_ID"),
    client_secret=os.getenv("NAVER_CLIENT_SECRET"),
)

query = input("검색어를 입력하세요: ")
url = client.build_url(query, display=5)
headers = client.build_headers()

response = requests.get(url, headers=headers)
client.raise_for_status(response)
news_list = client.parse_response(response.json())

for i, news in enumerate(news_list, 1):
    print(f"\n[{i}] {news['title']}")
    print(f"    링크: {news['link']}")
    print(f"    요약: {news['description'][:80]}...")
    
    print("    [본문 읽는 중...]")
    full_text = client.extract_full_text(news['link'])
    
    # 본문 전체를 출력합니다. 
    print(f"    --- 본문 시작 ---\n{full_text}\n    --- 본문 끝 ---\n")

