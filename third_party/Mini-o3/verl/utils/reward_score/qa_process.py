# verl/utils/reward_score/qa_process.py
"""
QA + 内网 GPT 结果打分（0/1） + GPT 过程打分（0~1）
- 结果打分：语义 judge（Score: 0/1）
- 过程打分：参考 iou_process 的多轮工具行为评分（human events + rollout + image）
- 两次 GPT 调用，完成不同任务
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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from .config import OPENAI_GATEWAY, GPT_RUNTIME

from functools import lru_cache

@lru_cache(maxsize=2048)
def _file_to_data_url_cached(image_abs_path: str) -> Optional[str]:
    return _file_to_data_url(image_abs_path)

# ============================================================
# 0) Prompts (Accuracy judge)
# ============================================================

ACC_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\n"
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully.\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

ACC_QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:

1. **Question Related to the Image**: {question}
2. **Ground Truth Answer**: {ground_truth}
3. **Model Predicted Answer**: {prediction}

Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:
- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?
- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:
(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.
(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.

**Output Format**:
Your response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect.
The format should be "Score: 0 or 1"
"""


# ============================================================
# 1) Process prompts (reuse iou_process style, but adapted to QA)
# ============================================================

_FLAG_LIST = [
    "converges_fast",
    "coarse_to_fine",
    "efficient_tool_use",
    "redundant_steps",
    "looping",
    "tool_call_loop",
    "random_search",
    "tool_misuse",
    "overlong_self_talk",
    "poor_convergence",
    "irrelevant_crops",
    "gibberish_or_repetition",
    "language_chaos",
]

PROCESS_PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "v1a": {
        "system": (
            "You are a strict process evaluator for multi-turn visual question answering with tool usage (crop/grounding).\n"
            "You will receive: (1) Human annotation events (reference), (2) Model multi-turn rollout (candidate), and an image.\n"
            "Score ONLY the model's process quality (tool usage + convergence), not final answer correctness.\n"
            "Return ONLY JSON with keys: score, reasons, flags.\n"
            "- score: float in [0,1]\n"
            "- reasons: <= 150 words\n"
            "- flags: a list chosen from the allowed set.\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),
        "instruction": (
            "Use human events as reference behavior.\n"
            "Score model process based on: efficiency, convergence, validity, and disciplined tool usage.\n"
            "Do NOT score final QA correctness."
        ),
    },
    "v1b": {
        "system": (
            "You are evaluating a reinforcement learning setup for multi-turn visual QA using tool calls.\n"
            "Context:\n"
            "- The model answers a question about an image.\n"
            "- During rollout, the model can use tool calls (crop/grounding) across multiple rounds.\n"
            "- We also have a human trajectory (events) that represents a natural and efficient visual exploration example.\n"
            "- The dataset consists of challenging visual search problems with small targets, numerous distractors, "
            "and high-resolution images. Most questions require careful visual inspection to answer correctly.\n\n"

            "Important:\n"
            "Human events are a reference signal for attention and exploration patterns.\n"
            "The model does NOT need to imitate the human trajectory exactly.\n"
            "Alternative efficient strategies are valid if they are evidence-driven and convergent.\n\n"

            "Inputs you will receive:\n"
            "A) Image (the scene)\n"
            "B) Human events: raw UI events during human visual exploration / annotation\n"
            "C) Model rollout: multi-turn assistant outputs (may include tool calls/grounding)\n\n"

            "Goal:\n"
            "Provide a PROCESS score in [0,1] measuring how effective and evidence-driven the model's multi-turn behavior is.\n"
            "Important: DO NOT evaluate final answer correctness; that is scored separately.\n\n"

            "Human events hints:\n"
            "- hover_move often indicates visual scanning.\n"
            "- slow movement or repeated hovering indicates focused attention.\n"
            "- long dwell in one region suggests a candidate region of interest.\n"
            "- zoom events indicate deliberate inspection.\n"
            "- Humans may focus on regions without zooming; those regions can still be valid crop targets.\n"
            "- The human trajectory represents one of the most concise and efficient exploration strategies for each question. "
            "If the model's rollout is significantly more complex than what the human needed (many more tool calls, "
            "much more self-talk), this is a strong signal of inefficiency or redundancy. "
            "The model should learn from the human exploration pattern.\n\n"

            "Output hygiene (communication quality):\n"
            "- Treat as degenerate communication (strongly penalize, typically 0.00~0.30): meaningless repeated characters, "
            "obvious gibberish, unreadable multilingual mixing, or reasoning that spins in circles without new evidence.\n"
            "- Use flags gibberish_or_repetition and/or language_chaos when applicable; use overlong_self_talk for excessive rambling.\n"
            "- Tool/behavior loops: repeated identical or near-identical <grounding> boxes with no progress, or many tool rounds "
            "without new visual evidence, or reasoning that restates the same dead-end conclusion — treat as looping or "
            "tool_call_loop and score low (often 0.20~0.45 unless the task truly requires many steps).\n\n"

            "Direct answer without tools (exception to strict no-zoom rules):\n"
            "- Some questions are answerable from the full image at native resolution without zooming.\n"
            "- If human events show little or no zoom / minimal exploration for that item, and the question clearly does not "
            "require spotting a tiny detail, a rollout with NO <grounding>, short coherent <redacted_thinking>, and a proper "
            "<answer> may score in the 0.50~0.75 range — NOT premature guessing.\n"
            "- Do NOT conflate this with visual-search items (small targets, fine text, distant objects): those still demand "
            "crop/grounding; no-tool rollouts there should stay low (<= 0.30).\n"
            "- When unsure, trust human events as the primary prior for how much exploration was appropriate.\n\n"

            "Process evaluation dimensions:\n"
            "1) Efficiency: minimal wasted steps; no repeated identical attempts.\n"
            "2) Convergence: moves from broad scan to focused crops (coarse-to-fine) when zoom is needed.\n"
            "3) Tool discipline: crops only when needed; avoids spamming; stops once confident.\n"
            "4) Evidence-driven exploration: actions are motivated by uncertainty or visual evidence needs.\n"
            "5) Communication discipline: avoids long self-talk; output stays readable and purposeful.\n\n"

            "Scoring rubric (MUST follow strictly):\n"
            "- 0.00~0.10: Degenerate output — empty or near-empty <think> block, immediate <answer> with no reasoning, "
            "repeated <answer> tags, or malformed output structure. Examples: '<think></think><answer>X</answer>', "
            "'<think><answer>X</answer><answer>X</answer>...'\n"
            "- 0.10~0.30: No tool use — model produces NO <grounding> tool calls and directly guesses an answer. "
            "Even if the model includes brief thinking text, skipping all visual exploration for a visual search task "
            "is premature guessing.\n"
            "- 0.30~0.50: Minimal/ineffective tool use — model uses tool calls but they are random, irrelevant, "
            "or repeatedly target the same region without progress.\n"
            "- 0.50~0.70: Partial exploration — some purposeful tool use but with notable inefficiencies: "
            "redundant crops, weak coarse-to-fine strategy, or not focusing on the question-relevant region.\n"
            "- 0.70~0.85: Good exploration — evidence-driven crops targeting relevant regions, reasonable convergence, "
            "minor inefficiencies only.\n"
            "- 0.85~1.00: Excellent — efficient coarse-to-fine strategy, each crop is purposeful and well-targeted, "
            "converges promptly once sufficient evidence is gathered, minimal self-talk.\n\n"

            "Critical rules:\n"
            "- This dataset contains challenging visual search problems. The vast majority of questions require "
            "crop/grounding tool use to inspect details. Do NOT assume answers are 'clearly visible' without zooming.\n"
            "- A model that produces NO <grounding> tool calls MUST score <= 0.30.\n"
            "- A model with empty/trivial <think> blocks and immediate <answer> MUST score <= 0.10.\n"
            "- Do NOT rationalize lack of tool use as 'efficient' or 'the question is simple'. "
            "If no tools are used, the score is low.\n"
            "- If the model's trajectory is much more complex than the human events "
            "(e.g. far more tool calls or excessive self-talk), penalize for inefficiency.\n"
            "- Do NOT assume shorter rollouts are better. Do NOT assume longer rollouts are better.\n"
            "- Reward the minimum exploration needed to obtain sufficient visual evidence.\n\n"

            "Convergence rule:\n"
            "- A trajectory is converged when the model performs meaningful exploration, "
            "then stops tool usage and produces a final answer.\n"
            "- If rollout ends without a final answer, treat it as poor convergence.\n"
            "- Skipping all exploration and immediately answering is NOT convergence — "
            "it is premature guessing.\n\n"

            "Output format:\n"
            "Return ONLY JSON:\n"
            "{\"score\": <float 0..1>, \"reasons\": <short string>, \"flags\": <list of tags>}\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),

        "instruction": (
            "Use human events as the primary prior for how much exploration and zooming was appropriate.\n"
            "Reward purposeful exploration and evidence-driven crops when the task requires them.\n"
            "Penalize looping, tool_call_loop, irrelevant crops, gibberish_or_repetition, language_chaos, and non-convergent behavior.\n"
            "Allow moderate scores (0.50~0.75) for concise no-tool answers only when the full image and human events support it; "
            "keep low scores for no-tool rollouts on true visual-search items.\n"
        ),
    },
    "v1c": {
        "system": (
            "You are an expert rater designing process reward for RL training of a multimodal model for visual QA.\n"
            "You must be CONSISTENT and strict.\n\n"
            "Rubric (process only, not answer correctness):\n"
            "- 0.90~1.00: very few steps; clear scan->focus->verify; crops are purposeful; stops promptly.\n"
            "- 0.70~0.89: mostly convergent; some redundancy.\n"
            "- 0.40~0.69: weak convergence; many redundant crops; noticeable self-talk.\n"
            "- 0.00~0.39: random search; frequent irrelevant crops; looping; no convergence.\n\n"
            "Return ONLY JSON with keys: score, reasons, flags.\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),
        "instruction": "Use human events as the desired reference trajectory. Score model process accordingly.",
    },
}


# ============================================================
# 2) OpenAI client + concurrency gates
# ============================================================

_client_lock = threading.Lock()
_client = None

# process gate (reusing GPT_RUNTIME knob)
_process_gate = threading.Semaphore(int(GPT_RUNTIME.get("max_concurrency", 8)))

# accuracy gate (separate knob)
def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

_acc_gate = threading.Semaphore(_env_int("QA_ACC_GPT_MAX_CONCURRENCY", 8))


class _RequestsChatCompletions:
    def __init__(self, api_key: str, base_url: str, verify_ssl: bool):
        import requests

        self._requests = requests
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._verify_ssl = verify_ssl

    def create(self, **kwargs):
        extra_headers = kwargs.pop("extra_headers", None) or {}
        extra_body = kwargs.pop("extra_body", None) or {}
        timeout = kwargs.pop("timeout", None)
        kwargs.pop("stream", None)
        env_extra_body = os.getenv("INTERNAL_OPENAI_EXTRA_BODY_JSON", "").strip()
        if env_extra_body:
            try:
                extra_body.update(json.loads(env_extra_body))
            except Exception as exc:
                raise ValueError(f"Invalid INTERNAL_OPENAI_EXTRA_BODY_JSON: {env_extra_body[:200]}") from exc
        kwargs.update(extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(extra_headers)
        resp = self._requests.post(
            self._url,
            headers=headers,
            json=kwargs,
            verify=self._verify_ssl,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = []
        for choice in data.get("choices", []):
            message = choice.get("message") or {}
            choices.append(SimpleNamespace(message=SimpleNamespace(content=message.get("content") or "")))
        return SimpleNamespace(choices=choices)


class _RequestsOpenAIClient:
    def __init__(self, api_key: str, base_url: str, verify_ssl: bool):
        self.chat = SimpleNamespace(completions=_RequestsChatCompletions(api_key, base_url, verify_ssl))


def _get_client():
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        credential = OPENAI_GATEWAY.get("api_key", "")
        base_url = OPENAI_GATEWAY.get("base_url", "")
        if not credential or not base_url:
            raise RuntimeError("Missing INTERNAL_OPENAI_API_KEY or INTERNAL_OPENAI_BASE_URL.")
        verify_ssl = bool(int(OPENAI_GATEWAY.get("verify_ssl", 1)))
        transport = str(OPENAI_GATEWAY.get("transport", "openai")).strip().lower()
        if transport in {"requests", "request"}:
            _client = _RequestsOpenAIClient(api_key=credential, base_url=base_url, verify_ssl=verify_ssl)
        else:
            import httpx
            from openai import OpenAI

            timeout_s = float(os.getenv("INTERNAL_OPENAI_HTTP_TIMEOUT_S", "60"))
            http_client = httpx.Client(verify=verify_ssl, timeout=timeout_s)
            _client = OpenAI(api_key=credential, base_url=base_url, http_client=http_client)
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
# 4) Helpers: extract final answer + format reward (optional)
# ============================================================

def _extract_answer_from_rollout(predict_str_list: Any, mode: str = "strict") -> Optional[str]:
    """
    从多轮输出中提取最终答案字符串（不含 <answer> 标签）
    mode:
      - split: 直接 split("<answer>") 取最后
      - strict: 正则找最后一个 <answer>...</answer>
      - strict_v2: 要求最后是 <answer>...</answer> 结尾
    """
    if isinstance(predict_str_list, list):
        text = " ".join([str(x) for x in predict_str_list])
    else:
        text = str(predict_str_list)

    if mode == "split":
        if "<answer>" not in text:
            return None
        return text.split("<answer>")[-1].split("</answer>")[0].strip()

    if mode == "strict":
        matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    if mode == "strict_v2":
        m = re.search(r"<answer>(.*?)</answer>\s*$", text, flags=re.DOTALL)
        if not m:
            return None
        return m.group(1).strip()

    raise ValueError(f"Unknown extract mode: {mode}")


def _format_ok(predict_str_list: Any) -> float:
    """
    QA baseline 先不强调 format，但如果你想要一个最轻量的格式约束，可用：
    - 最后必须包含 <answer>...</answer>
    """
    if isinstance(predict_str_list, list) and predict_str_list:
        last = str(predict_str_list[-1])
    else:
        last = str(predict_str_list)

    return 1.0 if (("<answer>" in last) and ("</answer>" in last)) else 0.0


# ============================================================
# 5) GPT call: QA accuracy judge (0/1)
# ============================================================

def _call_gpt_acc_score(question_text: str, predict_str_list: Any, ground_truth: str, extra_info: Dict[str, Any]) -> float:
    """
    返回 0.0 / 1.0
    """
    extract_mode = str(extra_info.get("extract_answer_tags", "strict"))
    gpt_extract_answer = bool(extra_info.get("gpt_extract_answer", True))

    pred = None
    if gpt_extract_answer:
        pred = _extract_answer_from_rollout(predict_str_list, mode=extract_mode)
        if pred is None:
            return 0.0
    else:
        # 直接用整个输出（不推荐，容易被多轮噪声污染）
        pred = " ".join(predict_str_list) if isinstance(predict_str_list, list) else str(predict_str_list)

    judge_prompt = ACC_QUERY_PROMPT.format(question=question_text, ground_truth=ground_truth, prediction=pred)

    model_name = OPENAI_GATEWAY.get("model", "gpt-4.1")
    max_retries = int(_env_int("QA_ACC_GPT_MAX_RETRIES", 3))
    backoff = float(os.getenv("QA_ACC_GPT_BACKOFF_S", "2"))
    timeout_s = float(os.getenv("QA_ACC_GPT_TIMEOUT_S", "60"))
    max_tokens = int(os.getenv("QA_ACC_GPT_MAX_TOKENS", "128"))

    # optional internal trace header
    extra_headers = {}
    trace_id = OPENAI_GATEWAY.get("trace_id", "")
    if trace_id:
        extra_headers["M-TraceId"] = trace_id

    client = _get_client()

    last_err = None
    for attempt in range(max_retries):
        try:
            with _acc_gate:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": ACC_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    stream=False,
                    extra_headers=extra_headers if extra_headers else None,
                    timeout=timeout_s,
                    temperature=min(0.2 * attempt, 1.0),
                    max_tokens=max_tokens,
                )
            text = (resp.choices[0].message.content or "").strip()
            # 解析 "Score: 0/1"
            if "score" not in text.lower():
                raise ValueError(f"bad judge output: {text[:200]}")
            token = text.lower().split("score")[-1]
            # 粗暴抽取 0/1（兼容 "Score: 1" / "Score:1" / "Score: 1\nReason..."）
            if "1" in token and "0" not in token:
                return 1.0
            if "0" in token and "1" not in token:
                return 0.0
            # 都在的话，取冒号后第一个数字
            m = re.search(r"score\s*:\s*([01])", text, flags=re.IGNORECASE)
            if m:
                return 1.0 if m.group(1) == "1" else 0.0
            raise ValueError(f"ambiguous judge output: {text[:200]}")

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))

    if int(os.getenv("QA_ACC_LOG", "1")) == 1:
        print(f"[qa_process][acc_gpt] mode=failed score=0 err={repr(last_err)}")
    return 0.0


def _parse_gpt_process_json_response(raw: str) -> Dict[str, Any]:
    """Parse process-judge JSON; tolerate markdown fences and extra surrounding text."""
    text = (raw or "").strip()
    m_fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m_fence:
        text = m_fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("no JSON object in process judge output", text, 0)


# ============================================================
# 6) GPT call: Process score (0~1), vision preferred + fallback
# ============================================================

def _call_gpt_process_score(question_text: str, human_events: Any, rollout_text: Any, sample_extra: Dict[str, Any]) -> float:
    variant = str(GPT_RUNTIME.get("prompt_variant", "v1b"))
    tmpl = PROCESS_PROMPT_TEMPLATES.get(variant, PROCESS_PROMPT_TEMPLATES["v1b"])

    doc_id = sample_extra.get("doc_id", "")
    payload = {
        "task_type": "visual_qa_with_tool_calls",
        "question": question_text,
        "doc_id": doc_id,
        "human_events": human_events,
        "model_rollout_round_text": rollout_text,
        "output_schema": {"score": "float[0,1]", "reasons": "string", "flags": f"subset of {_FLAG_LIST}"},
    }

    system_prompt = tmpl["system"]
    user_text = json.dumps(
        {"instruction": tmpl["instruction"], "payload": payload},
        ensure_ascii=False
    )

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
                #u = _file_to_data_url(abs_path)
                u = _file_to_data_url_cached(abs_path)
                if u:
                    img_data_urls.append(u)

    client = _get_client()
    model_name = OPENAI_GATEWAY.get("model", "gpt-4.1")

    max_retries = int(GPT_RUNTIME.get("max_retries", 3))
    backoff = float(GPT_RUNTIME.get("retry_backoff_s", 2))
    timeout_s = float(GPT_RUNTIME.get("timeout_s", 60))

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
            extra_headers=extra_headers if extra_headers else None,
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
            with _process_gate:
                try:
                    if use_vision and img_data_urls:
                        last_mode = "vision"
                        text = _run(with_images=True)
                    else:
                        last_mode = "text"
                        text = _run(with_images=False)
                except Exception as e_img:
                    vision_err = e_img
                    last_mode = "text_fallback"
                    text = _run(with_images=False)

            obj = _parse_gpt_process_json_response(text)
            score = float(obj.get("score"))
            score = max(0.0, min(1.0, score))

            if log_score:
                flags = obj.get("flags", [])
                reasons = (obj.get("reasons", "") or "").replace("\n", " ").strip()
                extra = ""
                if last_mode == "text_fallback" and vision_err is not None:
                    extra = f" vision_err={repr(vision_err)}"
                print(f"[qa_process][gpt_process] doc_id={doc_id} mode={last_mode} prompt={variant} score={score:.4f} reasons='{reasons}' flags={flags}{extra}")

            return score

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))

    if int(GPT_RUNTIME.get("log_gpt_score", 1)) == 1:
        print(f"[qa_process][gpt_process] doc_id={doc_id} mode=failed prompt={variant} score=0.0000 err={repr(last_err)}")
    return 0.0


# ============================================================
# 7) compute_score entrypoint
# ============================================================

def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    """
    返回 (total, acc_component, format_component)
    - acc_component：acc_w * acc
    - format_component：fmt_w * format_ok + process_score * acc * proc_w
    - 过程 GPT：仅当 process_reward_weight(proc_w) > 0 且 acc 判定为正确时调用
    """
    extra_info = extra_info or {}

    acc_w = float(extra_info.get("acc_reward_weight", 1.0))
    fmt_w = float(extra_info.get("format_reward_weight", 0.0))
    proc_w = float(extra_info.get("process_reward_weight", 0.0))

    sample_extra = extra_info.get("sample_extra_info") or {}
    
    # 1) accuracy (0/1) via GPT judge
    acc = _call_gpt_acc_score(question_text=prompt, predict_str_list=predict_str_list, ground_truth=ground_truth, extra_info=extra_info)
    acc_component = acc_w * float(acc)
    acc_is_correct = float(acc) >= 0.5

    # 2) (optional) format: minimal check
    fmt = _format_ok(predict_str_list)
    fmt_component = fmt_w * float(fmt)

    # 3) process (0~1) via GPT judge (vision preferred); only if proc_w>0 and correct (saves API)
    acc_val = float(acc)
    if proc_w > 0 and acc_is_correct:
        process_score = _call_gpt_process_score(
            question_text=prompt,
            human_events=sample_extra.get("events"),
            rollout_text=predict_str_list,
            sample_extra=sample_extra,
        )
    else:
        process_score = 0.0

    proc_component = float(process_score) * acc_val * proc_w

    format_component = fmt_component + proc_component
    total = acc_component + format_component
    return total, acc_component, format_component
