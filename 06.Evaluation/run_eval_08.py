#!/usr/bin/env python3
"""
run_eval_08.py — 08.MultiAgent RAG 평가 파이프라인

06.Evaluation 평가기(evaluators/)를 그대로 재사용하되
파이프라인 실행부를 08.MultiAgent/backend/rag.py 기반으로 교체한다.

Usage:
    python run_eval_08.py                            # 기본 (sample_testset.json)
    python run_eval_08.py --testset my_testset.json  # 커스텀 테스트셋
    python run_eval_08.py --no-llm                   # Faithfulness 평가 건너뜀
    python run_eval_08.py --no-rag                   # Coverage/Rule 평가만
    python run_eval_08.py --output reports/ma_run    # 출력 경로 지정
    python run_eval_08.py --faithfulness-mode single # 단일 점수 방식(빠름)
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
MA_DIR   = EVAL_DIR.parent / "08.MultiAgent" / "backend"

sys.path.insert(0, str(MA_DIR))

from dotenv import load_dotenv
load_dotenv(EVAL_DIR.parent / "08.MultiAgent" / ".env")

from openai import OpenAI
import rag
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


# ── RAG 파이프라인 실행 (08.MultiAgent 기반) ─────────────────

def run_pipeline(
    client: OpenAI,
    case: dict,
    chunks: list[dict],
    embeddings: list[list[float]],
    bm25,
    user_data: dict,
) -> dict:
    """
    testset 케이스 1개에 대해 감정 분류 → 미션 생성 → 검색 컨텍스트 재현.
    반환 dict는 각 Evaluator.evaluate()에 그대로 전달된다.

    testset 필드 매핑:
      input.emotion_text  → mood (rag 함수 인자)
      input.minutes       → minutes
      input.current_fruits → user_data["fruits"]
    """
    inp      = case.get("input", {})
    mood     = inp.get("emotion_text") or inp.get("mood", "")
    minutes  = inp.get("minutes", 15)
    time_str = f"{minutes}분"

    # current_fruits를 user_data에 반영 (중복 제거)
    eval_data = dict(user_data)
    if inp.get("current_fruits"):
        eval_data["fruits"] = list(inp["current_fruits"])

    # 1. 감정 분류
    emotion_type = rag.classify_emotion(client, mood)

    # 2. 미션 생성
    raw, is_wildcard, sources = rag.get_mission(
        client, mood, time_str, minutes,
        chunks, embeddings, emotion_type, eval_data, bm25=bm25,
    )
    parsed = rag.parse_mission(raw, is_wildcard, sources)

    # 3. 검색 컨텍스트 재현 (Retrieval / Faithfulness 평가용)
    if not is_wildcard and chunks:
        query      = rag.expand_query(mood, minutes, emotion_type=emotion_type)
        hyde_q     = rag.hyde_query(client, query)
        q_emb      = client.embeddings.create(
            model="text-embedding-3-small", input=[hyde_q]
        ).data[0].embedding
        top_chunks = rag.retrieve(
            q_emb, chunks, embeddings,
            k=5, emotion_type=emotion_type,
            query_text=hyde_q, bm25=bm25,
        )
        retrieved_sources = [c["source"] for c in top_chunks]
        retrieved_context = rag.build_context(top_chunks)
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
        "data_state":         eval_data,
        "combo_before":       eval_data.get("combo_count", 0),
        "combo_after":        eval_data.get("combo_count", 0),
    }


def run_pipeline_offline(case: dict, user_data: dict) -> dict:
    """RAG 없이 testset 내 precomputed 필드나 빈 값으로 구성."""
    inp = case.get("input", {})
    pre = case.get("precomputed") or {}

    eval_data = dict(user_data)
    if inp.get("current_fruits"):
        eval_data["fruits"] = list(inp["current_fruits"])

    return {
        "emotion_type":      pre.get("emotion_type", ""),
        "generated_mission": pre.get("mission", ""),
        "generated_basis":   pre.get("basis", ""),
        "generated_effect":  pre.get("effect", ""),
        "parsed_mission": {
            "mission":     pre.get("mission", ""),
            "category":    pre.get("category", ""),
            "difficulty":  pre.get("difficulty", "하"),
            "basis":       pre.get("basis", ""),
            "effect":      pre.get("effect", ""),
            "is_wildcard": pre.get("is_wildcard", False),
        },
        "is_wildcard":       pre.get("is_wildcard", False),
        "retrieved_sources": pre.get("retrieved_sources", []),
        "retrieved_context": pre.get("retrieved_context", ""),
        "retrieved_chunks":  [],
        "raw_response":      "",
        "data_state":        eval_data,
        "combo_before":      eval_data.get("combo_count", 0),
        "combo_after":       eval_data.get("combo_count", 0),
    }


# ── 점수 집계 / 리포트 (run_eval.py와 동일 로직) ─────────────

def aggregate_scores(all_results: list[dict]) -> dict:
    by_eval: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        for ev_r in r.get("eval_results", []):
            by_eval[ev_r["evaluator"]].append(ev_r["score"])

    summary = {}
    for ev, scores in by_eval.items():
        n = len(scores)
        summary[ev] = {
            "mean":      round(sum(scores) / n, 4),
            "min":       round(min(scores), 4),
            "max":       round(max(scores), 4),
            "pass_rate": round(sum(1 for s in scores if s >= 0.6) / n, 4),
            "n":         n,
        }
    return summary


def write_json_report(all_results, summary, output_dir, tag):
    report = {
        "generated_at": datetime.now().isoformat(),
        "target":       "08.MultiAgent",
        "summary":      summary,
        "cases":        all_results,
    }
    path = output_dir / f"report_ma_{tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown_report(all_results, summary, output_dir, tag):
    eval_order = ["retrieval", "faithfulness", "coverage", "rule"]
    lines = [
        "# RAG 평가 리포트 — 08.MultiAgent",
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
            "| 항목 | 값 |",
            "|---|---|",
            f"| 감정 분류 | {r.get('emotion_type', '-')} |",
            f"| 생성 미션 | {r.get('generated_mission', '-')} |",
            f"| 카테고리 / 난이도 | {r.get('category', '-')} / {r.get('difficulty', '-')} |",
            f"| 돌발 미션 | {'예 ⚡' if r.get('is_wildcard') else '아니오'} |",
        ]
        if r.get("retrieved_sources"):
            lines.append(f"| 검색 출처 (상위 3) | {' · '.join(r['retrieved_sources'][:3])} |")
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
            det    = ev_r.get("details", {})
            detail_str = ""
            if ev_r["evaluator"] == "retrieval" and det:
                parts = ["P@k: " + " / ".join(
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

    path = output_dir / f"report_ma_{tag}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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
        print(f"  {label:<22} {s['mean']:>6.3f}  {s['min']:>5.3f}  {s['max']:>5.3f}  {s['pass_rate']:>5.1%}")
    print(f"{'='*w}")


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="08.MultiAgent RAG 평가 파이프라인")
    parser.add_argument("--testset",           default="sample_testset.json")
    parser.add_argument("--output",            default="reports")
    parser.add_argument("--no-llm",            action="store_true")
    parser.add_argument("--no-rag",            action="store_true")
    parser.add_argument("--faithfulness-mode", choices=["claim", "single"], default="claim")
    parser.add_argument("--k-values",          default="1,3,5")
    args = parser.parse_args()

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
    cases   = testset if isinstance(testset, list) else testset.get("cases", [])
    print(f"✅ 테스트셋 로드: {testset_path.name}  ({len(cases)}개 케이스)")
    print(f"🎯 평가 대상: 08.MultiAgent  ({MA_DIR})")

    # ── OpenAI 클라이언트 ─────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 없습니다. 08.MultiAgent/.env 파일을 확인하세요.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # ── RAG 인덱스 ────────────────────────────────────────────
    user_data: dict = {
        "fruits": [], "cards": [], "combo_count": 0,
        "last_category": None, "mission_history": [],
        "weak_paper_boost": [],
    }
    if not args.no_rag:
        print("📚 RAG 인덱스 빌드 중...", end="", flush=True)
        chunks, embeddings, bm25 = rag.build_index(client)
        print(f" 완료 ({len(chunks)}개 청크)")
    else:
        chunks, embeddings, bm25 = [], [], None
        print("⚠️  --no-rag: RAG 인덱스 없이 실행 (Coverage·Rule 평가만 유효)")

    # ── 평가기 초기화 ─────────────────────────────────────────
    k_values   = [int(k) for k in args.k_values.split(",") if k.strip().isdigit()]
    evaluators = [RetrievalEvaluator(k_values=k_values)]
    if not args.no_llm:
        evaluators.append(FaithfulnessEvaluator(client, mode=args.faithfulness_mode))
    evaluators += [CoverageEvaluator(), RuleEvaluator()]

    # ── 케이스 실행 ───────────────────────────────────────────
    all_results: list[dict] = []

    for i, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{i:03d}")
        desc    = case.get("description", "")
        print(f"\n[{i}/{len(cases)}] {case_id}  {desc}")

        try:
            if args.no_rag:
                po = run_pipeline_offline(case, user_data)
            else:
                po = run_pipeline(client, case, chunks, embeddings, bm25, user_data)

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
    summary   = aggregate_scores(all_results)
    print_summary(summary)

    tag       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = write_json_report(all_results, summary, output_dir, tag)
    md_path   = write_markdown_report(all_results, summary, output_dir, tag)

    print(f"\n📄 JSON  리포트: {json_path.relative_to(EVAL_DIR)}")
    print(f"📝 Markdown 리포트: {md_path.relative_to(EVAL_DIR)}\n")


if __name__ == "__main__":
    main()
