# 06. RAG 평가 파이프라인

bloom_v10의 RAG 미션 생성 파이프라인을 자동으로 평가하는 프레임워크입니다.

---

## 디렉토리 구조

```
06.Evaluation/
├── run_eval.py              # 통합 실행 스크립트
├── sample_testset.json      # 예시 테스트셋 (5케이스)
├── testset_template.json    # 테스트셋 작성 템플릿 + 필드 설명
├── README.md
└── evaluators/
    ├── __init__.py
    ├── base.py              # EvalResult, BaseEvaluator
    ├── retrieval.py         # Retrieval 평가 (Precision@k, Recall@k, MRR)
    ├── faithfulness.py      # Faithfulness 평가 (LLM judge)
    ├── coverage.py          # Requirement Coverage 평가
    └── rule.py              # Rule 기반 구조 검사
```

---

## 평가 모듈

### 1. Retrieval 평가 (`RetrievalEvaluator`)

검색 품질을 측정합니다. 테스트케이스에 `relevant_doc_ids`가 있을 때만 동작합니다.

| 지표 | 설명 |
|------|------|
| `Precision@k` | 상위 k개 결과 중 관련 문서 비율 |
| `Recall@k` | 전체 관련 문서 중 상위 k개에 포함된 비율 |
| `Hit@k` | 상위 k개에 관련 문서가 하나라도 있으면 1 |
| `MRR` | Mean Reciprocal Rank — 첫 관련 문서의 순위 역수 |

- **기본 k값**: `[1, 3, 5]`
- **통과 기준**: `Precision@mid_k ≥ 0.33`
- `relevant_doc_ids`가 비어있으면 자동 스킵 (score=1.0)

---

### 2. Faithfulness 평가 (`FaithfulnessEvaluator`)

생성된 미션/근거가 검색된 논문 컨텍스트에 기반하는지 LLM이 판정합니다 (RAGAS 방식).

**claim 모드** (기본, 정확):
1. 생성 텍스트에서 검증 가능한 claim 최대 5개 추출
2. 각 claim이 검색 컨텍스트로 뒷받침되는지 YES/NO 판정
3. `score = 지지된 claim 수 / 전체 claim 수`

**single 모드** (빠름, 비용 절감):
- LLM이 0.0~1.0 점수를 직접 반환

- **통과 기준**: `score ≥ 0.6`
- 돌발 미션(is_wildcard=True)이고 컨텍스트가 없으면 score=1.0 자동 처리
- `--no-llm` 플래그 시 이 평가자는 건너뜀

---

### 3. Requirement Coverage 평가 (`CoverageEvaluator`)

테스트케이스에 명시된 요구사항 충족 비율을 측정합니다.

| 요구사항 키 | 검사 내용 |
|------------|----------|
| `mission_nonempty` | 미션 텍스트가 비어있지 않음 |
| `basis_nonempty` | 근거 텍스트가 비어있지 않음 |
| `effect_nonempty` | 효과 텍스트가 비어있지 않음 |
| `emotion_type_match` | 분류된 감정이 expected.emotion_type과 일치 |
| `category_allowed` | 미션 카테고리가 expected.category 목록 안에 있음 |
| `difficulty_allowed` | 난이도가 expected.difficulty 목록 안에 있음 |
| `time_feasible` | 난이도 기반 예상 소요 시간 ≤ input.minutes |
| `category_matches_emotion` | 감정 유형에 적합한 카테고리 (휴리스틱) |

- **통과 기준**: `score ≥ 0.8` (충족 비율 80% 이상)
- ground truth가 없는 항목(예: `emotion_type_match`인데 `expected.emotion_type` 없음)은 스킵

---

### 4. Rule 기반 평가 (`RuleEvaluator`)

LLM 없이 구조적 규칙을 즉시 검사합니다.

| 규칙 키 | 검사 내용 |
|--------|----------|
| `max_time_minutes` | 난이도 기반 예상 소요 시간 ≤ 값 |
| `allowed_categories` | 카테고리가 목록 안에 있음 |
| `allowed_difficulties` | 난이도가 목록 안에 있음 |
| `min_mission_length` | 미션 텍스트 길이 ≥ 값 (자) |
| `max_mission_length` | 미션 텍스트 길이 ≤ 값 (자) |
| `max_fruits` | 현재 fruits 개수 < 값 |
| `combo_wildcard_no_increment` | 돌발 미션이면 콤보 변화 없음 |
| `max_category_ratio` | fruits 중 단일 카테고리 비율 ≤ 값 |

- **통과 기준**: 모든 규칙 통과 (`score == 1.0`)

---

## 실행 방법

### 기본 실행 (전체 평가)

```bash
cd project/06.Evaluation
python run_eval.py --testset sample_testset.json
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--testset` | 테스트셋 JSON 파일 경로 | `sample_testset.json` |
| `--output` | 리포트 출력 디렉토리 | `reports/` |
| `--no-llm` | LLM 평가자(Faithfulness) 건너뜀 | False |
| `--no-rag` | RAG 인덱스 빌드 건너뜀 (오프라인) | False |
| `--faithfulness-mode` | `claim` 또는 `single` | `claim` |
| `--k-values` | Precision@k에 사용할 k 값들 | `1 3 5` |

### 오프라인 모드 (LLM 비용 없이 Coverage + Rule만)

```bash
python run_eval.py --testset sample_testset.json --no-llm --no-rag
```

테스트케이스에 `precomputed` 필드가 있으면 live 파이프라인을 호출하지 않고 미리 계산된 결과를 사용합니다.

### Faithfulness 단일 점수 모드 (빠른 평가)

```bash
python run_eval.py --testset sample_testset.json --faithfulness-mode single
```

---

## 출력 리포트

실행 후 `reports/` 디렉토리에 두 가지 파일이 생성됩니다:

### JSON 리포트 (`report_YYYYMMDD_HHMMSS.json`)

```json
{
  "meta": {
    "testset": "sample_testset.json",
    "timestamp": "2025-01-01T12:00:00",
    "total_cases": 5
  },
  "summary": {
    "retrieval":    {"mean": 0.8, "min": 0.6, "max": 1.0, "pass_rate": 0.8},
    "faithfulness": {"mean": 0.75, ...},
    "coverage":     {"mean": 0.9, ...},
    "rule":         {"mean": 0.95, ...}
  },
  "cases": [
    {
      "case_id": "case_001",
      "results": {
        "retrieval":    {"score": 0.8, "passed": true, ...},
        "faithfulness": {"score": 0.7, "passed": true, ...},
        "coverage":     {"score": 1.0, "passed": true, ...},
        "rule":         {"score": 1.0, "passed": true, ...}
      }
    }
  ]
}
```

### Markdown 리포트 (`report_YYYYMMDD_HHMMSS.md`)

케이스별 점수표와 평가자별 요약 통계를 포함합니다.

---

## 테스트셋 작성 가이드

### 최소 케이스 구성

```json
{
  "id": "case_001",
  "description": "케이스 설명",
  "input": {
    "emotion_text": "감정 문장",
    "minutes": 20,
    "current_fruits": []
  },
  "expected": {
    "emotion_type": "중립",
    "category": ["생산성", "성장"],
    "difficulty": ["하", "중"]
  },
  "requirements": ["mission_nonempty", "basis_nonempty", "time_feasible"],
  "rules": {
    "max_time_minutes": 20,
    "min_mission_length": 10
  },
  "relevant_doc_ids": [],
  "precomputed": null
}
```

전체 필드 설명은 [testset_template.json](testset_template.json)을 참고하세요.

### 오프라인 케이스 (`precomputed` 사용)

```json
{
  "id": "case_offline_001",
  "precomputed": {
    "parsed_mission": {
      "mission": "산책 10분 하기",
      "basis": "걷기 운동은 세로토닌 분비를 촉진한다",
      "effect": "기분 전환 및 스트레스 완화",
      "category": "건강",
      "difficulty": "하",
      "is_wildcard": false
    },
    "emotion_type": "부정적",
    "retrieved_context": "...",
    "retrieved_sources": [],
    "combo_before": 2,
    "combo_after": 3
  }
}
```

---

## 환경 설정

```
project/p1/.env 에 아래 키가 있어야 합니다:

OPENAI_API_KEY=sk-...
```

평가 스크립트는 `project/p1/.env`를 자동으로 로드합니다.

---

## 난이도별 예상 소요 시간 기준

| 난이도 | 예상 시간 |
|--------|---------|
| 하 | 5분 |
| 중 | 15분 |
| 상 | 30분 |
| 최상 | 60분 |
| 돌발 | 5분 |
| 도전 | 5분 |

`time_feasible` 요구사항과 `max_time_minutes` 규칙은 이 기준을 사용합니다.
