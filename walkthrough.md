# 🚀 전체 프로젝트 작업 내역 (Project Walkthrough)

지금까지 진행된 모든 AI 프로젝트와 시스템 구축 내역을 정리한 문서입니다. 이 문서는 GitHub에도 함께 저장되어 전체 작업의 흐름을 한눈에 파악할 수 있게 도와줍니다.

---

## 1. 💰 개인 맞춤형 "돈 되는 뉴스 필터" (Money News Filter)
**목표**: 단순한 뉴스 스크래핑을 넘어, 자산 가치에 영향을 줄 만한 핵심 정보를 추출합니다.

- **기술 스택**: Python, Naver Search API, OpenAI (GPT-4o), Trafilatura
- **핵심 로직**:
    1. **네이버 뉴스 수집**: 사용자가 입력한 키워드(예: AI 규제, 부동산 정책)로 가장 최신 뉴스를 가져옵니다.
    2. **본문 전문 추출**: 광고를 제외한 본문 글 내용만 깔끔하게 읽어옵니다.
    3. **AI 금융 분석**: GPT-4o가 기사를 읽고 "주가나 부동산 등에 실제로 영향이 있는지" 판별합니다.
    4. **영향도 요약**: 기사가 왜 중요한지 2-3줄로 명쾌하게 요약하여 제공합니다.
- **주요 파일**: `naver/run_money_filter.py`, `naver/news_analyzer.py`

---

## 2. 🧠 AI 의사결정 및 RAG 서비스 (Project P1)
**목표**: 외부 지식(PDF 등)을 활용해 AI가 현명한 결정을 내릴 수 있도록 돕는 시스템을 구축 중입니다.

- **기술 스택**: Streamlit, OpenAI API, PyPDF, RAG(Retrieval-Augmented Generation)
- **주요 기능**:
    - **Streamlit Web UI**: 사용자가 브라우저에서 바로 AI와 대화할 수 있는 인터페이스 제공.
    - **JSON/PDF 데이터 처리**: 인덱싱(`14.indexing.py`)을 통해 PDF 문서의 내용을 AI가 학습할 수 있도록 전처리.
    - **Decision AI**: 단순 챗봇을 넘어 상황을 분석하고 최선의 선택지를 제안하는 고도화된 로직(`decision_ai_v2.py`).
- **주요 파일**: `project/p1/Streamlit.py`, `project/p1/decision_ai_v2.py`

---

## 3. 🛡️ 개발 환경 구축 및 버전 관리 (Git & DevOps)
**목표**: 전문적인 개발 환경을 조성하고 협업 및 백업이 가능하게 합니다.

- **Git 통합**: 로컬 저장소 초기화 및 GitHub(`makejinjja/ax-curriculum`) 원격 연동 완료.
- **보안 강화**: `.gitignore` 설정을 통해 API Key(`.env`) 등 민감 정보가 유출되지 않도록 철저히 차단.
- **문서화**: 프로젝트별 README 및 전체 작업 워크스루(이 문서) 기록.

---

## 🎮 부록: 클래식 게임 프로젝트
- **Tetris Advance**: 기본적인 테트리스 기능을 넘어 AI 상대와 대결하거나 다양한 기능을 포함한 테트리스 엔진 구현.

---

**Last Updated**: 2026-04-21  
**Author**: Antigravity AI Assistant & makejinjja
