"""
Tool calling 接口 - 为 agentic 能力预留
注意: tool calling 必须用 /api/chat (非流式), 会牺牲 TTFA
仅在用户明确触发 agentic 命令时调用, 主对话路径不启用
"""
import json
import time

import requests


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time on the device.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vital_signs",
            "description": (
                "Get the user's latest heart rate (bpm) and breathing rate "
                "(breaths/min) from the mmWave radar. Use only when the user "
                "explicitly asks about their vitals or current physical state."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def dispatch_tool(name, args, radar):
    if name == "get_current_time":
        return json.dumps({"local_time": time.strftime("%H:%M:%S, %Y-%m-%d")})
    elif name == "get_vital_signs":
        s = radar.get_state()
        return json.dumps({
            "heart_rate_bpm": s.get("hr"),
            "breathing_rate_per_min": s.get("br"),
            "presence_detected": s.get("presence"),
            "inferred_state": s.get("category"),
        })
    return json.dumps({"error": f"unknown tool: {name}"})


def chat_with_tools(messages, model, ollama_chat_url, radar, max_iters=3):
    for _ in range(max_iters):
        r = requests.post(ollama_chat_url, json={
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }, timeout=60)
        msg = r.json().get("message", {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            fn = call["function"]["name"]
            args = call["function"].get("arguments", {}) or {}
            result = dispatch_tool(fn, args, radar)
            messages.append({"role": "tool", "name": fn, "content": result})

    return messages[-1].get("content", "(tool loop max iters exceeded)")


TOOL_KEYWORDS = (
    "what time", "current time",
    "my heart rate", "my breathing", "my vitals",
    "how am i", "physiological",
)


def should_use_tools(user_text):
    t = user_text.lower()
    return any(kw in t for kw in TOOL_KEYWORDS)
