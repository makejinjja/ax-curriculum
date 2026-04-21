import re
from urllib.parse import urlencode
import trafilatura

BASE_URL = "https://openapi.naver.com/v1/search/news.json"


class NaverAPIError(Exception):
    pass


class NaverNewsClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def build_url(self, query: str, display: int = 10, start: int = 1) -> str:
        return f"{BASE_URL}?{urlencode({'query': query, 'display': display, 'start': start, 'sort': 'date'})}"

    def build_headers(self) -> dict:
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    def raise_for_status(self, response) -> None:
        if response.status_code != 200:
            raise NaverAPIError(f"API 오류: {response.status_code}")

    def parse_response(self, response: dict) -> list:
        items = response["items"]
        for item in items:
            item["title"] = re.sub(r"<[^>]+>", "", item.get("title", ""))
            if "description" in item:
                item["description"] = re.sub(r"<[^>]+>", "", item["description"])
        return items

    def extract_full_text(self, url: str) -> str:
        """
        주어진 URL에서 뉴스 본문을 추출합니다.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                # trafilatura를 사용하여 본문만 추출
                text = trafilatura.extract(downloaded)
                return text if text else "본문 구문을 추출할 수 없습니다."
            return "페이지 다운로드 실패."
        except Exception as e:
            return f"본문 추출 중 오류 발생: {e}"
