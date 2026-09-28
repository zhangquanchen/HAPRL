# verl/utils/reward_score/iou_process.py
"""
V1：IoU + 内网 GPT 过程打分（默认传图 + 详细 prompt + fallback 日志）
- OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native")
- 默认 use_vision=1：优先发送图片；失败则回退 text-only 并打印 mode=text_fallback
- 不做缓存（你要求）
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import OPENAI_GATEWAY, GPT_RUNTIME


# ============================================================
# 1) bbox parsing + IoU
# ============================================================

_BBOX_RE = re.compile(r"<bbox>\s*([0-9eE\.\,\-\s]+)\s*</bbox>", flags=re.DOTALL)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(text, str) or not text:
        return None
    matches = _BBOX_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(" ", "")
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in parts]
    except Exception:
        return None
    x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)



# ============================================================
# 2) OpenAI client + concurrency gate
# ============================================================

_client_lock = threading.Lock()
_client = None
_gate = threading.Semaphore(int(GPT_RUNTIME.get("max_concurrency", 8)))


def _get_client():
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        from openai import OpenAI
        credential = OPENAI_GATEWAY.get("api_key", "")
        base_url = OPENAI_GATEWAY.get("base_url", "")
        if not credential or not base_url:
            raise RuntimeError("Missing INTERNAL_OPENAI_API_KEY or INTERNAL_OPENAI_BASE_URL.")
        _client = OpenAI(api_key=credential, base_url=base_url)
        return _client


# ============================================================
# 3) Image reading (BASE_IMAGE_DIR + relative path -> data URL)
# ============================================================

def _resolve_image_abs(rel_or_abs: str) -> Optional[str]:
    if not isinstance(rel_or_abs, str) or not rel_or_abs:
        return None
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    base_dir = os.environ.get("BASE_IMAGE_DIR", "")
    if not base_dir:
        return None
    return os.path.normpath(os.path.join(base_dir, rel_or_abs))


def _file_to_data_url(image_abs_path: str) -> Optional[str]:
    try:
        p = Path(image_abs_path)
        if not p.exists():
            return None
        mime, _ = guess_type(str(p))
        if mime is None:
            mime = "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# ============================================================
# 4) Prompt templates (more detailed & task-aligned)
# ============================================================

# flags 集合：后续可以统计分布
_FLAG_LIST = [
    "converges_fast",
    "coarse_to_fine",
    "efficient_tool_use",
    "redundant_steps",
    "looping",
    "random_search",
    "invalid_bbox_spam",
    "overlong_self_talk",
    "poor_convergence",
    "tool_misuse",
]

PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    # v1a：全面但简洁（适合先跑通并稳定）
    "v1a": {
        "system": (
            "You are a strict process evaluator for multi-turn visual localization with tool usage (crop/grounding).\n"
            "You will receive: (1) Human annotation events (reference), (2) Model multi-turn rollout (candidate), and an image.\n"
            "Your task is to score ONLY the model's process quality, not final IoU.\n"
            "Return ONLY JSON with keys: score, reasons, flags.\n"
            "- score: float in [0,1]\n"
            "- reasons: <= 150 words\n"
            "- flags: a list chosen from the allowed set.\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),
        "instruction": (
            "Use human events as reference behavior.\n"
            "Score model process based on: efficiency, convergence, validity, and disciplined tool usage.\n"
            "Do NOT score final bbox correctness."
        ),
    },

    # v1b：最贴合训练目标（推荐默认）
    "v1b": {
        "system": (
            "You are evaluating a reinforcement learning setup for bounding-box localization using multi-turn tool calls.\n"
            "Context:\n"
            "- The model must output a final normalized bbox (<answer><bbox>...</bbox></answer>)\n"
            "- During rollout, the model can use tool calls (crop/grounding) across multiple rounds.\n"
            "- We also have a human annotation trajectory (events) that represents a good/efficient labeling process.\n\n"
            "Inputs you will receive:\n"
            "A) Image (the scene)\n"
            "B) Human events: raw UI events during human bbox annotation\n"
            "C) Model rollout: multi-turn assistant outputs (may include tool calls/grounding)\n\n"
            "Goal:\n"
            "Provide a PROCESS score in [0,1] measuring how human-like and effective the model's multi-turn behavior is.\n"
            "Important: DO NOT evaluate final IoU/correctness; that is computed separately.\n\n"
            "Process evaluation dimensions (cover all):\n"
            "1) Efficiency: minimal wasted steps; no repeated identical attempts.\n"
            "2) Convergence: behavior should move from coarse localization to tight refinement (coarse-to-fine).\n"
            "3) Validity: avoids invalid/degenerate boxes; respects normalized bbox semantics; tool calls are well-formed.\n"
            "4) Tool discipline: uses crop only when needed; avoids spamming tool calls; stops once confident.\n"
            "5) Human-likeness: resembles rational annotator (scan -> focus -> refine), not random search.\n"
            "6) Communication discipline: avoids long self-talk; avoids excessive repeated reasoning.\n\n"
            "Output format:\n"
            "Return ONLY JSON:\n"
            "{\"score\": <float 0..1>, \"reasons\": <short string>, \"flags\": <list of tags>}\n"
            f"Allowed flags: {_FLAG_LIST}\n"
            "Your score should be consistent across samples.\n"
        ),
        "instruction": (
            "Compare model rollout to human events as reference.\n"
            "If model exhibits coarse-to-fine refinement with few purposeful tool calls, score high.\n"
            "If model loops, spams invalid bboxes, or behaves like random search, score low.\n"
        ),
    },

    # v1c：严格 rubric（后期更稳定、可解释）
    "v1c": {
        "system": (
            "You are an expert rater designing process reward for RL training of a multimodal model that outputs bounding boxes.\n"
            "You must be CONSISTENT and strict.\n\n"
            "Rubric (process only, not IoU):\n"
            "- 0.90~1.00: very few steps; clear coarse-to-fine refinement; almost no invalid boxes; tool calls are purposeful; stops promptly.\n"
            "- 0.70~0.89: mostly convergent; some redundancy or minor invalid attempts.\n"
            "- 0.40~0.69: weak convergence; many redundant attempts; noticeable self-talk; some tool misuse.\n"
            "- 0.00~0.39: random search; frequent invalid bbox spam; looping; no convergence.\n\n"
            "Return ONLY JSON with keys: score, reasons, flags.\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),
        "instruction": (
            "Use human events as the desired reference trajectory. Score model process accordingly."
        ),
    },
}


# ============================================================
# 5) GPT call (prefer vision; fallback text-only with explicit logs)
# ============================================================

def _call_gpt_process_score(human_events: Any, rollout_text: Any, sample_extra: Dict[str, Any]) -> float:
    variant = GPT_RUNTIME.get("prompt_variant", "v1b")
    tmpl = PROMPT_TEMPLATES.get(variant, PROMPT_TEMPLATES["v1b"])

    doc_id = sample_extra.get("doc_id", "")
    payload = {
        "instruction": tmpl["instruction"],
        "doc_id": doc_id,
        "annotation_time": sample_extra.get("annotation_time"),
        "task_evaluation": sample_extra.get("task_evaluation"),
        "human_events": human_events,
        "model_rollout_round_text": rollout_text,
        "output_schema": {"score": "float[0,1]", "reasons": "string", "flags": f"subset of {_FLAG_LIST}"},
    }

    system_prompt = tmpl["system"]
    user_text = json.dumps(payload, ensure_ascii=False)

    use_vision = int(GPT_RUNTIME.get("use_vision", 1)) == 1
    max_images = int(GPT_RUNTIME.get("max_images_to_gpt", 1))
    image_detail = GPT_RUNTIME.get("image_detail", "auto")
    log_score = int(GPT_RUNTIME.get("log_gpt_score", 1)) == 1

    # image data urls
    img_data_urls: List[str] = []
    imgs = sample_extra.get("images")
    if use_vision and isinstance(imgs, list) and len(imgs) > 0:
        for rel_or_abs in imgs[:max_images]:
            abs_path = _resolve_image_abs(rel_or_abs)
            if abs_path:
                u = _file_to_data_url(abs_path)
                if u:
                    img_data_urls.append(u)

    client = _get_client()
    model_name = OPENAI_GATEWAY.get("model", "gpt-4.1")

    max_retries = int(GPT_RUNTIME.get("max_retries", 3))
    backoff = float(GPT_RUNTIME.get("retry_backoff_s", 2))
    timeout_s = float(GPT_RUNTIME.get("timeout_s", 60))

    # optional internal trace header
    extra_headers = {}
    trace_id = OPENAI_GATEWAY.get("trace_id", "")
    if trace_id:
        extra_headers["M-TraceId"] = trace_id

    def _run(with_images: bool) -> str:
        if with_images and img_data_urls:
            content = [{"type": "text", "text": user_text}]
            for u in img_data_urls:
                content.append({"type": "image_url", "image_url": {"url": u, "detail": image_detail}})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            extra_headers=extra_headers,
            timeout=timeout_s,
            temperature=0.0,
            max_tokens=350,
        )
        return resp.choices[0].message.content or ""

    last_mode = "text"
    last_err = None
    vision_err = None

    for attempt in range(max_retries):
        try:
            with _gate:
                try:
                    # ✅ default: prefer vision
                    if use_vision and img_data_urls:
                        last_mode = "vision"
                        text = _run(with_images=True)
                    else:
                        last_mode = "text"
                        text = _run(with_images=False)
                except Exception as e_img:
                    # vision failed -> fallback
                    vision_err = e_img
                    last_mode = "text_fallback"
                    text = _run(with_images=False)

            obj = json.loads(text)
            score = float(obj.get("score"))
            score = max(0.0, min(1.0, score))

            if log_score:
                flags = obj.get("flags", [])
                extra = ""
                if last_mode == "text_fallback" and vision_err is not None:
                    extra = f" vision_err={repr(vision_err)}"
                #print(f"[iou_process][gpt] doc_id={doc_id} mode={last_mode} prompt={variant} score={score:.4f} flags={flags}{extra}")
                reasons = obj.get("reasons", "").replace("\n", " ").strip()  # 处理换行，避免日志错乱
                print(f"[iou_process][gpt] doc_id={doc_id} mode={last_mode} prompt={variant} score={score:.4f} reasons='{reasons}' flags={flags}{extra}")

            return score

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))

    if int(GPT_RUNTIME.get("log_gpt_score", 1)) == 1:
        print(f"[iou_process][gpt] doc_id={doc_id} mode=failed prompt={variant} score=0.0000 err={repr(last_err)}")
    return 0.0


# ============================================================
# 6) compute_score entrypoint
# ============================================================

def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    extra_info = extra_info or {}
    acc_w = float(extra_info.get("acc_reward_weight", 1.0))
    fmt_w = float(extra_info.get("format_reward_weight", 0.0))
    proc_w = float(extra_info.get("process_reward_weight", 0.0))

    sample_extra = extra_info.get("sample_extra_info") or {}

    pred_text = predict_str_list[-1] if isinstance(predict_str_list, list) and predict_str_list else str(predict_str_list)
    pred_bbox = _parse_bbox_from_text(pred_text)
    gt_bbox = _parse_bbox_from_text(ground_truth)

    try:
        if pred_bbox is None or gt_bbox is None:
            iou = 0.0
            fmt = 0.0
        else:
            iou = _iou(pred_bbox, gt_bbox)
            fmt = 1.0
    except Exception as e:
        # 任何异常都不影响训练，直接当作 iou=0
        iou = 0.0
        fmt = 0.0


    acc_component = acc_w * float(iou)
    fmt_component = fmt_w * float(fmt)

    if proc_w > 0:
        process_score = _call_gpt_process_score(
            human_events=sample_extra.get("events"),
            rollout_text=predict_str_list,
            sample_extra=sample_extra,
        )
    else:
        process_score = 0.0

    proc_component = proc_w * float(process_score)
    format_component = fmt_component + proc_component
    total = acc_component + format_component
    return total, acc_component, format_component
