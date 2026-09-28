# verl/utils/reward_score/config.py
"""
内网 GPT 网关配置（sankuai AIGC）
- 使用 OpenAI SDK: OpenAI(api_key=..., base_url=...)
- 默认启用传图（use_vision=1）
- 过程 judge 的 prompt 版本：IOU_PROCESS_PROMPT_VARIANT
  - qa_process / iou_process：v1a | v1b | v1c（见各模块 PROCESS_PROMPT_TEMPLATES）
  - qa_process_v1：使用模板键 v1_text，请设 v1_text；其它值会回退到 v1_text
"""

from __future__ import annotations
import os


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v not in (None, "") else default


# -----------------------------
# Internal OpenAI Gateway config
# -----------------------------
OPENAI_GATEWAY = {
    "api_key": _env("INTERNAL_OPENAI_API_KEY", ""),
    "base_url": _env("INTERNAL_OPENAI_BASE_URL", "https://aigc.sankuai.com/v1/openai/native"),
    "model": _env("INTERNAL_OPENAI_MODEL", "gpt-4.1"),
    "verify_ssl": int(_env("INTERNAL_OPENAI_VERIFY_SSL", "1")),
    "transport": _env("INTERNAL_OPENAI_TRANSPORT", "openai"),
    # Optional header for internal tracing
    "trace_id": _env("INTERNAL_OPENAI_TRACE_ID", ""),
}

# -----------------------------
# Runtime knobs (NO CACHE)
# -----------------------------
GPT_RUNTIME = {
    "max_concurrency": int(_env("IOU_PROCESS_GPT_MAX_CONCURRENCY", "8")),
    "max_retries": int(_env("IOU_PROCESS_GPT_MAX_RETRIES", "3")),
    "retry_backoff_s": float(_env("IOU_PROCESS_GPT_BACKOFF_S", "2")),
    "timeout_s": float(_env("IOU_PROCESS_GPT_TIMEOUT_S", "60")),

    # qa_process_v1 请用 v1_text；未设置时默认 v1b 以兼容 qa_process / iou_process
    "prompt_variant": _env("IOU_PROCESS_PROMPT_VARIANT", "v1b"),

    # ✅ 默认强制传图（你要求）
    # 1: try vision; if it fails -> fallback text-only with explicit log
    "use_vision": int(_env("IOU_PROCESS_USE_VISION", "1")),
    "max_images_to_gpt": int(_env("IOU_PROCESS_MAX_IMAGES", "1")),
    "image_detail": _env("IOU_PROCESS_IMAGE_DETAIL", "auto"),

    # ✅ 是否打印 GPT 打分日志
    "log_gpt_score": int(_env("IOU_PROCESS_LOG_GPT_SCORE", "1")),
    "process_max_tokens": int(_env("IOU_PROCESS_MAX_TOKENS", "700")),
}
