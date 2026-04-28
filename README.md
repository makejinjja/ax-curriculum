# Project 폴더 문서

## 폴더 구조

```
project/
├── main.py                        # 뉴스 필터 프로젝트 (초기 실습)
├── data/
│   ├── knowledge.txt.pdf          # RAG용 행동 지식베이스 (구조화된 행동 목록)
│   └── inf.pdf.pdf                # RAG용 심리·신경과학 이론 지식베이스
└── p1/                            # Decision Removal AI 프로젝트
    ├── .env                       # API 키 설정 (OPENAI_API_KEY)
    ├── requirements.txt           # 패키지 목록
    ├── .index_cache/
    │   └── inf_index.json         # 임베딩 캐시 (자동 생성)
    ├── RAG.py                     # v1 — knowledge.txt.pdf 기반 RAG
    ├── decision_ai_no_rag.py      # v2 — RAG 없는 순수 GPT 버전
    ├── decision_ai.py             # v3 — inf.pdf 기반 RAG + 상세 근거 출력
    ├── decision_ai_v2.py          # v4 — 고도화 RAG (메타데이터 필터 + 캐시)
    ├── Streamlit.py               # v3 웹 UI 버전 (Streamlit)
    └── 14.indexing.py             # 인덱싱 파이프라인 고도화 실험 + 전후 비교
```

---

## 프로젝트 개요

**Decision Removal AI** — 사용자의 현재 상태(기분, 에너지, 시간, 목표)를 입력받아 "지금 당장 해야 할 행동 딱 1개"를 GPT가 결정해주는 시스템. 선택지를 없애서 행동을 유도하는 것이 핵심 목적.

---

## 파일별 설명

### `main.py`
뉴스 필터 프로젝트의 초기 실습 파일. 사용자가 키워드를 입력하면 해당 키워드 기반으로 뉴스를 가져오는 흐름의 입력 처리 부분. 뉴스 수집 기능은 미구현 상태.

---

### `data/knowledge.txt.pdf`
RAG.py에서 사용하는 지식베이스. 행동 목록이 구조화된 형태로 저장되어 있음.

형식:
```
[Action] 행동명
[Category] 카테고리
[Best For] 적합 조건
[Why Now] 이유
[Expected Outcome] 예상 결과
[How] 실행법
[Keywords] 키워드
==========
```

---

### `data/inf.pdf.pdf`
decision_ai.py / decision_ai_v2.py에서 사용하는 심리·신경과학 이론 지식베이스. 4개 탭으로 구성.

| 탭 | 내용 |
|---|---|
| 탭1 | 각성-동기 상태 모델 (무기력형 / 스트레스형 / 안정형 / 집중형) |
| 탭2 | 개인 최적화 전략 유형 (건강 / 생산성 / 돈 / 균형 중심형) |
| 탭3 | 에너지 상태 분류 (저하 / 균형 / 활성) |
| 탭4 | Circadian Rhythm 기반 시간대별 최적화 (7개 시간 슬롯) |

---

### `p1/RAG.py` — v1
**knowledge.txt.pdf**를 지식베이스로 쓰는 첫 번째 RAG 버전.

- 지식베이스: `knowledge.txt.pdf` (행동 목록)
- 청킹 방식: `=====` 구분자로 행동 단위 분리
- 검색: 전체 벡터 코사인 유사도
- 출력: 행동 / 이유 / 실행 방법 / 소요 시간

---

### `p1/decision_ai_no_rag.py` — v2
RAG 없이 GPT만으로 동작하는 버전. 사용자 상태를 그대로 프롬프트에 넣어서 결정.

- 지식베이스: 없음
- 검색: 없음
- 출력: 행동 / 이유 / 실행 방법 / 소요 시간
- 용도: RAG 적용 전후 품질 비교용 베이스라인

---

### `p1/decision_ai.py` — v3
**inf.pdf**를 지식베이스로 쓰는 RAG 버전. 신경과학 이론 근거를 출력에 포함.

- 지식베이스: `inf.pdf.pdf` (심리·신경과학 이론)
- 청킹 방식: 탭 → 섹션 단위 flat 분리
- 검색: 전체 벡터 코사인 유사도 (필터 없음)
- 쿼리 확장: 기분/목표/에너지 → 신경과학 키워드로 변환
- 출력: 행동 / 심리·생리 상태 분석 / 근거(상태이론·목표연계·시간) / 실행 방법 / 소요 시간

---

### `p1/decision_ai_v2.py` — v4 ✅ 최신
**14.indexing.py**의 고도화 파이프라인을 실제 서비스에 적용한 버전.

- 지식베이스: `inf.pdf.pdf`
- 청킹 방식: DocType별 전용 파서 4개 (탭마다 다른 로직)
- 검색: 메타데이터 사전필터 → 벡터 검색 → DocType 다양성 보장
- 인덱싱: SHA-256 해시 캐시 (`.index_cache/inf_index.json`) — 2회차부터 즉시 로드
- 출력: v3와 동일 (행동 / 분석 / 근거 / 실행 방법 / 소요 시간)
- 실행: `python decision_ai_v2.py` / 강제 재빌드: `python decision_ai_v2.py --rebuild`

v3 대비 개선점:

| 항목 | v3 (decision_ai.py) | v4 (decision_ai_v2.py) |
|---|---|---|
| 청킹 | 단일 flat 파서 | DocType별 전용 파서 4개 |
| 메타데이터 | 4개 필드 | mood/energy/goal/time/arousal 등 |
| 검색 | 전체 벡터 | 사전필터 → 벡터 → 다양성 보장 |
| 인덱싱 속도 | 매번 재임베딩 | SHA-256 캐시로 2회차부터 즉시 |

---

### `p1/Streamlit.py` — 웹 UI
decision_ai.py (v3)를 Streamlit 웹 인터페이스로 구현한 버전.

- 디자인: 블랙 & 화이트 / Inter 폰트
- 기능: 사이드바에서 상태 입력 → 결과 카드 형태로 시각화
- 인덱싱: `@st.cache_resource`로 최초 1회만 임베딩 실행
- 실행: `streamlit run Streamlit.py`

---

### `p1/14.indexing.py` — 인덱싱 파이프라인 실험
v4에 적용된 고도화 기법을 독립적으로 구현하고 **전후 비교 데모**를 보여주는 실험용 파일.

주요 기능:
1. **DocType별 파서** — 탭마다 도메인 특화 청킹 전략 적용
2. **구조 보존 전처리** — 소제목(특징/신경학적 상태/요약 등) 별도 추출
3. **메타데이터 확장** — mood_tags, energy_tags, goal_tags, time_start/end, arousal_level, keywords
4. **증분 인덱싱** — SHA-256 해시로 PDF 변경 여부 확인, 변경 없으면 캐시 재사용

실행하면 출력되는 것:
- BEFORE: 기존 flat 방식 청크 수 / 검색 결과
- AFTER: 고도화 방식 청크 수 / 메타데이터 필터 후보 / 검색 결과
- 개선 효과 요약표

실행: `python 14.indexing.py`

---

## 설치 및 실행

```bash
# 패키지 설치
pip install -r p1/requirements.txt

# .env 파일에 API 키 설정
echo "OPENAI_API_KEY=your_key_here" > p1/.env

# CLI 버전 실행 (최신)
python p1/decision_ai_v2.py

# 웹 UI 실행
streamlit run p1/Streamlit.py
```

---

## 개발 흐름 (버전 히스토리)

```
main.py              → 뉴스 필터 초기 실습
RAG.py               → RAG 첫 구현 (knowledge.txt 기반)
decision_ai_no_rag.py → RAG 제거, 순수 GPT 베이스라인
decision_ai.py        → inf.pdf 기반 RAG + 상세 신경과학 근거
Streamlit.py          → 웹 UI 추가
14.indexing.py        → 인덱싱 고도화 실험
decision_ai_v2.py     → 고도화 파이프라인 실서비스 적용
```
