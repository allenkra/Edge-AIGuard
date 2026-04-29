"""
基于雷达状态的 system prompt 模板
HR/BR 数值在每轮对话开始前实时注入 (radar.get_state() 是 O(1))
"""

BASE_PERSONA = (
    "You are a helpful voice assistant running locally on a Raspberry Pi. "
    "You have access to the user's real-time physiological signals from a "
    "60GHz mmWave radar (heart rate, breathing rate, presence)."
)

STATE_INSTRUCTIONS = {
    "normal": (
        "The user appears calm and relaxed. "
        "Keep answers brief and natural, under 2 sentences."
    ),
    "stressed": (
        "The user's elevated heart rate and breathing suggest stress or anxiety. "
        "Speak gently and slowly. Keep answers very brief, under 2 sentences. "
        "If appropriate, you may gently suggest taking a deep breath. "
        "Do NOT proactively mention specific numbers or cause alarm — "
        "but if the user explicitly asks about their vitals, share them calmly."
    ),
    "relaxed": (
        "The user appears very relaxed. Match their calm, unhurried energy. "
        "Up to 3 sentences is fine."
    ),
    "exercising": (
        "The user appears physically active. "
        "Give one-sentence answers only. Be direct and clear."
    ),
    "no_user": (
        "No user is currently detected. Respond briefly."
    ),
    "unknown": (
        "Keep answers brief and natural, under 2 sentences."
    ),
}

READING_RULE = (
    "You may share specific HR/BR values if the user explicitly asks "
    "(e.g., 'what is my heart rate?'). Do not bring them up unprompted "
    "unless directly relevant to the user's question."
)


def build_system_prompt(radar_state):
    category = radar_state.get("category", "unknown")
    style = STATE_INSTRUCTIONS.get(category, STATE_INSTRUCTIONS["unknown"])

    hr = radar_state.get("hr")
    br = radar_state.get("br")
    presence = radar_state.get("presence", False)

    if hr is not None and br is not None:
        readings = (
            f"Current readings: heart rate {hr:.0f} bpm, "
            f"breathing rate {br:.0f}/min, "
            f"presence={'yes' if presence else 'no'}, "
            f"inferred state: {category}."
        )
    else:
        readings = f"Physiological readings unavailable. State: {category}."

    # Order: static → category-conditioned → fully dynamic numerics last.
    # Putting HR/BR at the very end maximizes KV-cache prefix reuse for
    # speculative LLM prefill — see test_kv_cache.py scenario 7.
    return f"{BASE_PERSONA}\n\n{style}\n\n{READING_RULE}\n\n{readings}"


if __name__ == "__main__":
    for cat in ["normal", "stressed", "relaxed", "exercising", "no_user"]:
        fake = {"hr": 72, "br": 16, "presence": True, "category": cat}
        print(f"=== {cat} ===")
        print(build_system_prompt(fake))
        print()
