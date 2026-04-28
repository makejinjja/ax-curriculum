import os
import requests
from dotenv import load_dotenv
from naver_news import NaverNewsClient
from news_analyzer import NewsAnalyzer

def main():
    load_dotenv()
    
    # 1. 클라이언트 초기화
    try:
        naver_client = NaverNewsClient(
            client_id=os.getenv("NAVER_CLIENT_ID"),
            client_secret=os.getenv("NAVER_CLIENT_SECRET"),
        )
        analyzer = NewsAnalyzer()
    except Exception as e:
        print(f"초기화 오류: {e}")
        return

    print("=" * 60)
    print(" 🔎 개인 맞춤형 돈 되는 뉴스 필터 🔎")
    print("=" * 60)
    query = input("관심 키워드를 입력하세요 (예: AI 규제, 금리 인상, 전기차): ")
    print(f"\n[{query}] 관련 최신 뉴스를 수집하고 분석 중입니다. 잠시만 기다려주세요...\n")
    
    # 2. 네이버 뉴스 가져오기 (분석풀을 위해 기본 10개 수집)
    url = naver_client.build_url(query, display=10, start=1)
    headers = naver_client.build_headers()
    
    try:
        response = requests.get(url, headers=headers)
        naver_client.raise_for_status(response)
        news_list = naver_client.parse_response(response.json())
    except Exception as e:
        print(f"뉴스 수집 중 오류: {e}")
        return
    
    # 3. 뉴스 분석 및 필터링
    impactful_news = []
    
    for i, news in enumerate(news_list, 1):
        print(f"[{i:02d}/10] 분석 중: {news['title']}")
        
        # 3.1. 본문 추출
        full_text = naver_client.extract_full_text(news['link'])
        
        # 3.2. 본문이 너무 짧거나 오류일 경우 패스
        if not full_text or len(full_text.strip()) < 50 or "오류 발생" in full_text:
            continue
            
        # 3.3. OpenAI를 이용해 수익성 여부 판별
        analysis_result = analyzer.analyze(title=news['title'], full_text=full_text)
        
        if analysis_result:
            impactful_news.append({
                "title": news['title'],
                "link": news['link'],
                "area": analysis_result["area"],
                "summary": analysis_result["summary"]
            })
            
    # 4. 결과 출력
    print("\n\n" + "="*60)
    print(f" 💰 '{query}' 기반 돈 되는 뉴스 필터링 결과 💰")
    print("="*60)
    
    if not impactful_news:
        print("\n최근 10개의 최신 뉴스 중 직접적인 자산 가치 변동이나")
        print("돈과 관련된 의미 있는 변화를 가져올 만한 뉴스는 없습니다.")
        print("\n💡 팁: '금리', '반도체 보조금', '부동산 대책' 등 자본과 직결된 검색어를 추천합니다!")
    else:
        for idx, item in enumerate(impactful_news, 1):
            print(f"\n{idx}. {item['title']}")
            print(f"   📍 영향 분야 : {item['area']}")
            print(f"   💡 영 향 요 약 : {item['summary']}")
            print(f"   🔗 링 크 : {item['link']}")
            print("-" * 60)

if __name__ == "__main__":
    main()
