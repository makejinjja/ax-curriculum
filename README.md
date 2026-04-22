# ax-curriculum

Claude AI를 활용한 파이썬 학습 프로젝트 모음입니다.

---

## 폴더 구조

```
ax-workspace/
├── tetris.py                    # 테트리스 게임 (pygame 기반)
├── tetris_project/              # 테트리스 리팩터링 버전
├── 01_tetris/                   # 테트리스 초기 구현
├── 02_tetris_advance/           # 테트리스 고도화 버전
├── 03_ax_curriculum_chatbot/    # AX 커리큘럼 챗봇
├── 04.RAG/                      # RAG 학습 실험
├── naver/                       # 네이버 뉴스 필터 (초기 실습)
├── project/                     # Decision Removal AI + 기분전환 미션 프로젝트
│   ├── README.md                # 프로젝트 세부 문서
│   ├── main.py                  # 뉴스 필터 초기 실습
│   ├── data/                    # RAG 지식베이스 PDF
│   └── p1/                      # 핵심 AI 파일들
└── main.py                      # 루트 메인 (뉴스 필터)
```

---

## 프로젝트별 설명

### 01 / 02 테트리스
pygame을 이용한 테트리스 게임. 초기 구현 → 고도화(다음 블록 미리보기, 레벨업, 스코어보드) 순서로 발전.

- 실행: `python tetris.py`

---

### 03 AX 커리큘럼 챗봇
OpenAI API를 활용한 AX 커리큘럼 Q&A 챗봇.

---

### 04 RAG 실험
Retrieval-Augmented Generation 기초 실험. PDF → 청킹 → 임베딩 → 검색 → GPT 응답 파이프라인 학습.

---

### project/ — Decision Removal AI + 기분전환 미션 AI

이 폴더의 핵심 프로젝트. 두 가지 시스템으로 구성됨.

#### Decision Removal AI
사용자의 현재 상태(기분·에너지·시간·목표)를 입력받아 **"지금 당장 해야 할 행동 딱 1개"** 를 결정해주는 시스템. 선택지를 없애 행동을 유도하는 것이 목적.

| 파일 | 설명 |
|---|---|
| `p1/RAG.py` | v1 — knowledge.txt 기반 RAG |
| `p1/decision_ai_no_rag.py` | v2 — RAG 없는 순수 GPT 베이스라인 |
| `p1/decision_ai.py` | v3 — inf.pdf 기반 RAG + 신경과학 근거 |
| `p1/decision_ai_v2.py` | v4 — DocType별 파서 + 메타데이터 필터 + 캐시 ✅ 최신 |
| `p1/Streamlit.py` | v3 웹 UI 버전 |
| `p1/14.indexing.py` | 인덱싱 파이프라인 고도화 실험 |

```bash
python project/p1/decision_ai_v2.py         # CLI 실행
streamlit run project/p1/Streamlit.py       # 웹 UI 실행
```

#### 기분전환 미션 AI
기분과 가용 시간을 입력하면 GPT가 랜덤 미션을 제안하고, 완료 시 **열매 나무 + 카드 수집** 게임으로 보상을 주는 시스템.

| 파일 | 설명 |
|---|---|
| `p1/decision_mood_mission.py` | CLI 버전 (ANSI 컬러 나무 + 카드) |
| `p1/mood_mission_app.py` | Streamlit 웹 UI 버전 (SVG 나무 시각화) |

```bash
python project/p1/decision_mood_mission.py  # CLI 실행
streamlit run project/p1/mood_mission_app.py  # 웹 UI 실행
```

**게임 규칙**
- 미션 수락 → 완료 → 열매 획득 (난이도별 색상: 🔴하 🟤중 ⚪상 🟡최상)
- 나무에 열매 10개 도달 시 미션 진행 불가 → 열매 쪼개기로 카드 변환 필요
- 카드 종류: 체력 포션 / 목재 / 다람쥐 / 골드

---

## 설치 및 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt
pip install -r project/p1/requirements.txt

# API 키 설정
cp .env.example project/p1/.env
# project/p1/.env 파일에 OPENAI_API_KEY 입력
```

---

## 개발 흐름

```
테트리스 게임              → pygame 기초
네이버 뉴스 필터           → API 호출 기초
RAG.py                   → RAG 첫 구현
decision_ai_no_rag.py    → GPT 단독 베이스라인
decision_ai.py           → RAG + 신경과학 지식베이스
14.indexing.py           → 인덱싱 파이프라인 고도화 실험
decision_ai_v2.py        → 고도화 파이프라인 실서비스 적용
decision_mood_mission.py → 기분전환 미션 + 게임화 (CLI)
mood_mission_app.py      → Streamlit 웹 UI 버전
```
