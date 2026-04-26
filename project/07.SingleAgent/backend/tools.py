"""
tools.py — The 4 callable tools exposed to the ReAct agent

  1. rag_search        — BM25+cosine hybrid over internal knowledge base
  2. web_search        — Tavily real-time web search
  3. generate_curriculum — GPT-4o-mini structured JSON curriculum generation
  4. validate_curriculum — deterministic rule-based validation (time/structure/group)
"""
from __future__ import annotations
import json
import os
from typing import Any

from openai import OpenAI

import rag as rag_module


# ── 1. RAG search ─────────────────────────────────────────────

def rag_search(client: OpenAI, query: str, k: int = 5) -> str:
    try:
        idx = rag_module.get_index(client)
        results = idx.search(client, query, k=k)
        if not results:
            return "No relevant documents found in the AX Compass knowledge base."
        parts = [
            f"[{i+1}] Source: {r['source']} (score={r['score']})\n{r['text']}"
            for i, r in enumerate(results)
        ]
        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        return f"RAG search error: {exc}"


# ── 2. Web search ─────────────────────────────────────────────

def web_search(query: str) -> str:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return "Web search unavailable: TAVILY_API_KEY not configured."
    try:
        from tavily import TavilyClient
        res = TavilyClient(api_key=key).search(
            query=query, max_results=5, search_depth="basic"
        )
        parts = [
            f"[{r.get('title', 'No title')}]\n{r.get('content', '')}\nURL: {r.get('url', '')}"
            for r in res.get("results", [])
        ]
        return "\n\n---\n\n".join(parts) if parts else "No results found."
    except Exception as exc:
        return f"Web search error: {exc}"


# ── 3. Generate curriculum ────────────────────────────────────

def generate_curriculum(
    client: OpenAI,
    topic: str,
    target_audience: str,
    total_duration_minutes: int,
    group_size: str,
    learning_objectives: list[str],
    additional_context: str = "",
) -> dict[str, Any]:
    obj_list = "\n".join(f"- {o}" for o in learning_objectives)
    prompt = f"""You are a professional curriculum designer following the AX Compass methodology.

Design a complete, facilitation-ready curriculum with these specifications:

Topic: {topic}
Target Audience: {target_audience}
Total Duration: {total_duration_minutes} minutes
Group Size: {group_size}
Learning Objectives:
{obj_list}
Additional Context: {additional_context if additional_context else "None"}

Rules:
- Module durations MUST sum to exactly {total_duration_minutes} minutes
- Every module MUST have at least one concrete activity
- Use Bloom's Taxonomy action verbs in objectives
- Activities must be specific and facilitation-ready (not vague)
- Minimum 2 modules

Return ONLY a valid JSON object (no markdown fences):
{{
  "title": "...",
  "target_audience": "...",
  "total_duration_minutes": {total_duration_minutes},
  "group_size": "...",
  "learning_objectives": ["...", "..."],
  "modules": [
    {{
      "title": "...",
      "duration_minutes": <integer>,
      "objectives": ["..."],
      "activities": ["detailed activity description"],
      "materials": ["..."]
    }}
  ],
  "assessment": "...",
  "notes": "..."
}}"""

    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content)


# ── 4. Validate curriculum ────────────────────────────────────

def validate_curriculum(curriculum: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    # Required fields
    required = [
        "title", "target_audience", "total_duration_minutes",
        "group_size", "learning_objectives", "modules",
    ]
    for field in required:
        if field not in curriculum:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "score": 0.0}

    total      = curriculum["total_duration_minutes"]
    modules    = curriculum["modules"]
    group_size = str(curriculum.get("group_size", "")).lower()

    # ── Time rule ─────────────────────────────────────────────
    module_sum = sum(m.get("duration_minutes", 0) for m in modules)
    if abs(module_sum - total) > 5:
        errors.append(
            f"Module durations sum to {module_sum} min but declared total is {total} min "
            f"(tolerance ±5 min). Adjust module durations."
        )

    # ── Session length rules ──────────────────────────────────
    for m in modules:
        dur  = m.get("duration_minutes", 0)
        name = m.get("title", "Unnamed")
        if dur > 90:
            warnings.append(
                f"Module '{name}' is {dur} min — consider splitting or adding a break."
            )
        if dur < 5:
            warnings.append(f"Module '{name}' is only {dur} min — may feel rushed.")

    # ── Structure rules ───────────────────────────────────────
    if len(modules) < 2:
        warnings.append(
            "Only 1 module detected. Add at least an introduction and wrap-up."
        )

    for m in modules:
        name = m.get("title", "Unnamed")
        if not m.get("objectives"):
            warnings.append(f"Module '{name}' has no learning objectives.")
        if not m.get("activities"):
            errors.append(
                f"Module '{name}' has no activities — curriculum is not actionable."
            )

    if len(curriculum.get("learning_objectives", [])) < 2:
        warnings.append(
            "Fewer than 2 overall learning objectives. Add more specificity."
        )

    if not curriculum.get("assessment"):
        warnings.append("No assessment method defined — add how learning will be measured.")

    # ── Group rules ───────────────────────────────────────────
    large_group = any(kw in group_size for kw in ["large", "50+", "100", "200"])
    for m in modules:
        acts = " ".join(m.get("activities", [])).lower()
        if large_group and "pair" in acts and "break" not in m.get("title", "").lower():
            warnings.append(
                f"Module '{m.get('title')}': pair activities in a large group "
                "require clear facilitation instructions."
            )

    # ── Score ─────────────────────────────────────────────────
    score = max(0.0, min(1.0, 1.0 - len(errors) * 0.3 - len(warnings) * 0.07))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "score": round(score, 2),
    }
