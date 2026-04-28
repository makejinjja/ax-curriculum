#!/usr/bin/env python3
"""
run_eval.py — RAG 평가 파이프라인 통합 실행 스크립트

Usage:
    python run_eval.py                                  # 기본 (sample_testset.json)
    python run_eval.py --testset my_testset.json        # 커스텀 테스트셋
    python run_eval.py --no-llm                         # Faithfulness 평가 건너뜀
    python run_eval.py --no-rag                         # RAG 인덱스 없이 Coverage/Rule만
    python run_eval.py --output reports/my_run          # 출력 경로 지정
    python run_eval.py --faithfulness-mode single       # 단일 점수 방식(빠름)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
P1_DIR   = EVAL_DIR.parent / "p1"

sys.path.insert(0, str(P1_DIR))

from dotenv import load_dotenv
load_dotenv(P1_DIR / ".env")

from openai import OpenAI

import bloom_v10 as bloom  # p1/bloom_v10.py

from evaluators import (
    RetrievalEvaluator,
    FaithfulnessEvaluator,
    CoverageEvaluator,
    RuleEvaluator,
)

EVAL_LABELS = {
    "retrieval":    "Retrieval (Precision@k)",
    "faithfulness": "Faithfulness",
    "coverage":     "Requirement Coverage",
    "rule":         "Rule-based",
}


# ── RAG 파이프라인 실행 ───────────────────────────────────────

def run_pipeline(
    client: OpenAI,
    case: dict,
    chunks: list[dict],
    embeddings: list[list[float]],
    bm25,
    data: dict,
) -> dict:
    """
    케이스 1개에 대해 감정 분류 → 미션 생성 → 검색 컨텍스트 재현까지 실행.
    반환 dict는 각 Evaluator의 evaluate()에 그대로 전달된다.
    """
    inp      = case.get("input", {})
    mood     = inp.get("mood", "")
    minutes  = inp.get("minutes", 15)
    time_str = f"{minutes}분"

    # 1. 감정 분류
    emotion_type = bloom.classify_emotion(client, mood)

    # 2. 미션 생성
    raw, is_wildcard, sources = bloom.get_mission(
        client, mood, time_str, minutes,
        chunks, embeddings, emotion_type, data, bm25=bm25,
    )
    parsed = bloom.parse_mission(raw, is_wildcard, sources)

    # 3. 검색 컨텍스트 재현 (Retrieval / Faithfulness 평가용)
    if not is_wildcard and chunks:
        query    = bloom._expand_query(mood, minutes, emotion_type=emotion_type)
        hyde_q   = bloom._hyde_query(client, query)
        q_emb    = client.embeddings.create(
            model="text-embedding-3-small", input=[hyde_q]
        ).data[0].embedding
        top_chunks = bloom.retrieve(
            q_emb, chunks, embeddings,
            k=5, emotion_type=emotion_type,
            query_text=hyde_q, bm25=bm25,
        )
        retrieved_sources = [c["source"] for c in top_chunks]
        retrieved_context = bloom.build_context(top_chunks)
    else:
        top_chunks        = []
        retrieved_sources = []
        retrieved_context = ""

    return {
        "emotion_type":       emotion_type,
        "generated_mission":  parsed.get("mission", ""),
        "generated_basis":    parsed.get("basis", ""),
        "generated_effect":   parsed.get("effect", ""),
        "parsed_mission":     parsed,
        "is_wildcard":        is_wildcard,
        "retrieved_sources":  retrieved_sources,
        "retrieved_context":  retrieved_context,
        "retrieved_chunks":   top_chunks,
        "raw_response":       raw,
        "data_state":         data,
        "combo_before":       data.get("combo_count", 0),
        "combo_after":        data.get("combo_count", 0),
    }


# ── 오프라인 파이프라인 (no-rag 모드) ────────────────────────

def run_pipeline_offline(case: dict, data: dict) -> dict:
    """RAG 없이 testset 내 precomputed 필드나 빈 값으로 pipeline_output 구성."""
    pre = case.get("precomputed", {})
    return {
        "emotion_type":       pre.get("emotion_type", ""),
        "generated_mission":  pre.get("mission", ""),
        "generated_basis":    pre.get("basis", ""),
        "generated_effect":   pre.get("effect", ""),
        "parsed_mission": {
            "mission":     pre.get("mission", ""),
            "category":    pre.get("category", ""),
            "difficulty":  pre.get("difficulty", "하"),
            "basis":       pre.get("basis", ""),
            "effect":      pre.get("effect", ""),
            "is_wildcard": pre.get("is_wildcard", False),
        },
        "is_wildcard":        pre.get("is_wildcard", False),
        "retrieved_sources":  pre.get("retrieved_sources", []),
        "retrieved_context":  pre.get("retrieved_context", ""),
        "retrieved_chunks":   [],
        "raw_response":       "",
        "data_state":         data,
        "combo_before":       data.get("combo_count", 0),
        "combo_after":        data.get("combo_count", 0),
    }


# ── 점수 집계 ─────────────────────────────────────────────────

def aggregate_scores(all_results: list[dict]) -> dict:
    by_eval: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        for ev_r in r.get("eval_results", []):
            by_eval[ev_r["evaluator"]].append(ev_r["score"])

    summary = {}
    for ev, scores in by_eval.items():
        n = len(scores)
        summary[ev] = {
            "mean":        round(sum(scores) / n, 4),
            "min":         round(min(scores), 4),
            "max":         round(max(scores), 4),
            "pass_rate":   round(sum(1 for s in scores if s >= 0.6) / n, 4),
            "n":           n,
        }
    return summary


# ── JSON 리포트 ───────────────────────────────────────────────

def write_json_report(
    all_results: list[dict],
    summary: dict,
    output_dir: Path,
    tag: str,
) -> Path:
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary":      summary,
        "cases":        all_results,
    }
    path = output_dir / f"report_{tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Markdown 리포트 ───────────────────────────────────────────

def write_markdown_report(
    all_results: list[dict],
    summary: dict,
    output_dir: Path,
    tag: str,
) -> Path:
    eval_order = ["retrieval", "faithfulness", "coverage", "rule"]
    lines = [
        "# RAG 평가 리포트",
        f"\n> 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
        "## 📊 요약 점수\n",
        "| 평가 항목 | 평균 점수 | 최소 | 최대 | Pass Rate | 케이스 수 |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for ev in eval_order:
        if ev not in summary:
            continue
        s     = summary[ev]
        label = EVAL_LABELS.get(ev, ev)
        lines.append(
            f"| {label} | **{s['mean']:.3f}** | {s['min']:.3f} | {s['max']:.3f} "
            f"| {s['pass_rate']:.1%} | {s['n']} |"
        )
    lines += ["\n---\n", "## 🔍 케이스별 결과\n"]

    for r in all_results:
        case_id = r.get("case_id", "-")
        desc    = r.get("description", "")
        err     = r.get("error")

        lines.append(f"### `{case_id}`")
        if desc:
            lines.append(f"> {desc}\n")

        if err:
            lines.append(f"⚠️ **오류**: `{err}`\n")
            continue

        lines += [
            f"| 항목 | 값 |",
            f"|---|---|",
            f"| 감정 분류 | {r.get('emotion_type', '-')} |",
            f"| 생성 미션 | {r.get('generated_mission', '-')} |",
            f"| 카테고리 / 난이도 | {r.get('category', '-')} / {r.get('difficulty', '-')} |",
            f"| 돌발 미션 | {'예 ⚡' if r.get('is_wildcard') else '아니오'} |",
        ]
        if r.get("retrieved_sources"):
            srcs = r["retrieved_sources"][:3]
            lines.append(f"| 검색 출처 (상위 3) | {' · '.join(srcs)} |")

        lines += [
            "",
            "**평가 결과**\n",
            "| 평가 항목 | 점수 | 통과 | 세부 정보 |",
            "|---|:---:|:---:|---|",
        ]
        for ev_r in r.get("eval_results", []):
            label  = EVAL_LABELS.get(ev_r["evaluator"], ev_r["evaluator"])
            status = "✅" if ev_r["passed"] else "❌"
            notes  = ev_r.get("notes", "") or ""
            # Retrieval 세부 지표 inline
            detail_str = ""
            det = ev_r.get("details", {})
            if ev_r["evaluator"] == "retrieval" and det:
                parts = [f"P@k: " + " / ".join(
                    f"@{k}={det.get(f'precision@{k}', '-')}"
                    for k in [1, 3, 5] if f"precision@{k}" in det
                )]
                if "mrr" in det:
                    parts.append(f"MRR={det['mrr']}")
                detail_str = ", ".join(parts)
            elif notes:
                detail_str = notes
            lines.append(f"| {label} | {ev_r['score']:.3f} | {status} | {detail_str} |")
        lines.append("")

    path = output_dir / f"report_{tag}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── 콘솔 요약 출력 ────────────────────────────────────────────

def print_summary(summary: dict):
    w = 52
    print(f"\n{'='*w}")
    print(f"  {'평가 항목':<22} {'평균':>6}  {'최소':>5}  {'최대':>5}  {'통과율':>6}")
    print(f"{'─'*w}")
    for ev in ["retrieval", "faithfulness", "coverage", "rule"]:
        if ev not in summary:
            continue
        s     = summary[ev]
        label = EVAL_LABELS.get(ev, ev)[:22]
        print(
            f"  {label:<22} {s['mean']:>6.3f}  {s['min']:>5.3f}  {s['max']:>5.3f}  {s['pass_rate']:>5.1%}"
        )
    print(f"{'='*w}")


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="bloom RAG 평가 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--testset", default="sample_testset.json",
        help="테스트셋 JSON 경로 (기본: sample_testset.json)",
    )
    parser.add_argument(
        "--output", default="reports",
        help="리포트 출력 디렉토리 (기본: reports/)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Faithfulness LLM 평가 건너뜀 (API 비용 절약)",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="RAG 인덱스 빌드 없이 실행 (Coverage/Rule 평가만 유효)",
    )
    parser.add_argument(
        "--faithfulness-mode", choices=["claim", "single"], default="claim",
        help="Faithfulness 평가 방식: claim(정확) / single(빠름)",
    )
    parser.add_argument(
        "--k-values", default="1,3,5",
        help="Retrieval Precision@k 값 (쉼표 구분, 기본: 1,3,5)",
    )
    args = parser.parse_args()

    # ── 출력 디렉토리 ─────────────────────────────────────────
    output_dir = EVAL_DIR / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 테스트셋 로드 ─────────────────────────────────────────
    testset_path = Path(args.testset)
    if not testset_path.is_absolute():
        testset_path = EVAL_DIR / testset_path
    if not testset_path.exists():
        print(f"❌ 테스트셋 파일을 찾을 수 없습니다: {testset_path}")
        sys.exit(1)

    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    if isinstance(testset, list):
        cases = testset
    else:
        cases = testset.get("cases", [])
    print(f"✅ 테스트셋 로드: {testset_path.name}  ({len(cases)}개 케이스)")

    # ── OpenAI 클라이언트 ─────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 없습니다. p1/.env 파일을 확인하세요.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # ── RAG 인덱스 ────────────────────────────────────────────
    data = bloom.load_data()
    if not args.no_rag:
        print("📚 RAG 인덱스 빌드 중...", end="", flush=True)
        chunks, embeddings, bm25 = bloom.build_index(client)
        print(f" 완료 ({len(chunks)}개 청크)")
    else:
        chunks, embeddings, bm25 = [], [], None
        print("⚠️  --no-rag: RAG 인덱스 없이 실행 (Coverage·Rule 평가만 유효)")

    # ── 평가기 초기화 ─────────────────────────────────────────
    k_values = [int(k) for k in args.k_values.split(",") if k.strip().isdigit()]
    evaluators = [RetrievalEvaluator(k_values=k_values)]
    if not args.no_llm:
        evaluators.append(
            FaithfulnessEvaluator(client, mode=args.faithfulness_mode)
        )
    evaluators += [CoverageEvaluator(), RuleEvaluator()]

    # ── 케이스 실행 ───────────────────────────────────────────
    all_results: list[dict] = []

    for i, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{i:03d}")
        desc    = case.get("description", "")
        print(f"\n[{i}/{len(cases)}] {case_id}  {desc}")

        try:
            if args.no_rag:
                po = run_pipeline_offline(case, data)
            else:
                po = run_pipeline(client, case, chunks, embeddings, bm25, data)

            ev_results = []
            for ev in evaluators:
                er = ev.evaluate(case, po)
                ev_results.append(er.to_dict())
                status = "✅" if er.passed else "❌"
                print(f"  {status} {ev.name:<15} score={er.score:.3f}"
                      + (f"  [{er.notes}]" if er.notes else ""))

            all_results.append({
                "case_id":           case_id,
                "description":       desc,
                "emotion_type":      po.get("emotion_type", ""),
                "generated_mission": po.get("generated_mission", ""),
                "category":          po.get("parsed_mission", {}).get("category", ""),
                "difficulty":        po.get("parsed_mission", {}).get("difficulty", ""),
                "is_wildcard":       po.get("is_wildcard", False),
                "retrieved_sources": po.get("retrieved_sources", []),
                "eval_results":      ev_results,
            })

        except Exception as e:
            print(f"  ⚠️  오류: {e}")
            all_results.append({
                "case_id":      case_id,
                "description":  desc,
                "error":        str(e),
                "eval_results": [],
            })

    # ── 집계 + 리포트 ─────────────────────────────────────────
    summary = aggregate_scores(all_results)
    print_summary(summary)

    tag       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = write_json_report(all_results, summary, output_dir, tag)
    md_path   = write_markdown_report(all_results, summary, output_dir, tag)

    print(f"\n📄 JSON  리포트: {json_path.relative_to(EVAL_DIR)}")
    print(f"📝 Markdown 리포트: {md_path.relative_to(EVAL_DIR)}\n")


if __name__ == "__main__":
    main()
