"""
agent.py — OpenAI tool-calling ReAct agent loop

Flow per turn:
  1. Build messages (system + history + user)
  2. Call OpenAI with TOOL_DEFS
  3. If tool_calls → execute, append results, loop
  4. If no tool_calls → final answer, return ChatResponse
  5. Max MAX_ITERATIONS total steps; max MAX_VALIDATION_RETRIES failed validations
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

import tools as tool_mod
from schemas import ChatMessage, ChatResponse, ValidationResult

MAX_ITERATIONS        = 12
MAX_VALIDATION_RETRIES = 3

# ── Tool schema definitions ───────────────────────────────────

TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the internal AX Compass curriculum knowledge base for examples, "
                "design methodologies, templates, and best practices. "
                "Use this FIRST to ground the curriculum in real examples before generating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "English search query about curriculum design topics",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information about a specific topic, "
                "industry trends, or subject matter not covered in the knowledge base. "
                "Use when the topic requires up-to-date or niche external knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_curriculum",
            "description": (
                "Generate a complete, structured curriculum JSON. "
                "Only call this once you have confirmed ALL of: "
                "topic, target_audience, total_duration_minutes, group_size, learning_objectives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Main subject or skill the curriculum covers",
                    },
                    "target_audience": {
                        "type": "string",
                        "description": "Who will attend (role, experience level, background)",
                    },
                    "total_duration_minutes": {
                        "type": "integer",
                        "description": "Total available time in minutes",
                    },
                    "group_size": {
                        "type": "string",
                        "description": "Expected number of participants (e.g. '10-20 people')",
                    },
                    "learning_objectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-5 specific, measurable learning objectives using Bloom's verbs",
                    },
                    "additional_context": {
                        "type": "string",
                        "description": "Any extra requirements, constraints, or preferences",
                        "default": "",
                    },
                },
                "required": [
                    "topic", "target_audience", "total_duration_minutes",
                    "group_size", "learning_objectives",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_curriculum",
            "description": (
                "Validate a curriculum against time, structure, and group rules. "
                "ALWAYS call this immediately after generate_curriculum. "
                "If validation fails, fix the issues and call generate_curriculum again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "curriculum": {
                        "type": "object",
                        "description": "The curriculum object returned by generate_curriculum",
                    },
                },
                "required": ["curriculum"],
            },
        },
    },
]


# ── System prompt loader ──────────────────────────────────────

def _load_system_prompt() -> str:
    p = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
    return p.read_text(encoding="utf-8") if p.exists() else (
        "You are an expert curriculum design assistant. Help users create structured learning curricula."
    )


# ── Tool executor ─────────────────────────────────────────────

def _exec_tool(client: OpenAI, name: str, args: dict[str, Any]) -> str:
    if name == "rag_search":
        return tool_mod.rag_search(client, **args)
    if name == "web_search":
        return tool_mod.web_search(**args)
    if name == "generate_curriculum":
        result = tool_mod.generate_curriculum(client, **args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    if name == "validate_curriculum":
        result = tool_mod.validate_curriculum(**args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    return f"Unknown tool: {name}"


# ── Main agent loop ───────────────────────────────────────────

def run_agent(
    client: OpenAI,
    user_message: str,
    history: list[ChatMessage],
) -> ChatResponse:
    # Build initial message list
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _load_system_prompt()}
    ]
    messages.extend({"role": m.role, "content": m.content} for m in history)
    messages.append({"role": "user", "content": user_message})

    calls_made:   list[str]              = []
    curriculum:   dict[str, Any] | None  = None
    val_result:   ValidationResult | None = None
    val_attempts: int                    = 0
    complete:     bool                   = False

    for _iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=TOOL_DEFS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Append assistant turn (tool_calls may be present)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls — agent has reached a final answer
        if not msg.tool_calls:
            if curriculum and val_result and val_result.valid:
                complete = True
            return ChatResponse(
                reply=msg.content or "",
                complete=complete,
                curriculum=curriculum,
                validation_result=val_result,
                tool_calls_made=calls_made,
            )

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            calls_made.append(name)

            tool_result = _exec_tool(client, name, args)

            # Track domain state
            if name == "generate_curriculum":
                try:
                    curriculum = json.loads(tool_result)
                except json.JSONDecodeError:
                    pass

            elif name == "validate_curriculum":
                try:
                    vd = json.loads(tool_result)
                    val_result = ValidationResult(**vd)
                    if not val_result.valid:
                        val_attempts += 1
                        if val_attempts < MAX_VALIDATION_RETRIES:
                            tool_result += (
                                f"\n\n🔁 Validation failed (attempt {val_attempts}/{MAX_VALIDATION_RETRIES}). "
                                "Fix all ERRORS listed above and call generate_curriculum again with corrections."
                            )
                        else:
                            tool_result += (
                                f"\n\n⚠️ Maximum validation retries ({MAX_VALIDATION_RETRIES}) reached. "
                                "Present the current curriculum to the user with a note about the issues."
                            )
                    else:
                        complete = True
                except Exception:
                    pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # Fallback: max iterations reached
    return ChatResponse(
        reply=(
            "최대 처리 단계에 도달했습니다. "
            "요청을 더 구체적으로 다시 시도해 주세요."
        ),
        complete=False,
        curriculum=curriculum,
        validation_result=val_result,
        tool_calls_made=calls_made,
    )
