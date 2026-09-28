# verl/utils/reward_score/qa_process_v1.py
"""
QA reward with:
- GPT accuracy judge (0/1), reused from qa_process
- GPT process judge (0~1), adapted to textual human reference trajectories

Compared with qa_process.py, this version assumes sample_extra_info["events"]
is already a Mini-o3-style textual process template rather than raw mouse logs.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from .qa_process import (
    OPENAI_GATEWAY,
    GPT_RUNTIME,
    _call_gpt_acc_score,
    _env_int,
    _file_to_data_url_cached,
    _format_ok,
    _get_client,
    _process_gate,
    _resolve_image_abs,
)


_FLAG_LIST = [
    "converges_fast",
    "coarse_to_fine",
    "efficient_tool_use",
    "redundant_steps",
    "looping",
    "random_search",
    "tool_misuse",
    "overlong_self_talk",
    "poor_convergence",
    "irrelevant_crops",
    "target_miss",
    "weak_final_evidence",
    "broad_only",
    "progressive_refinement",
    "useful_reset",
    "redundant_recheck",
    "oversearch_after_evidence",
    "invalid_tool_json",
    "wrong_source_view",
    "looping_without_gain",
    "concise_convergence",
    "sufficient_no_crop_evidence",
    "char_spam",
    "language_drift",
]

_DEFAULT_TASK_PROFILE = "general_search"

_COMPONENT_KEYS = [
    "target_semantics",
    "evidence_acquisition",
    "search_progress",
    "tool_discipline",
    "communication_discipline",
]

_DEFAULT_COMPONENT_WEIGHTS = {
    "target_semantics": 0.20,
    "evidence_acquisition": 0.30,
    "search_progress": 0.25,
    "tool_discipline": 0.15,
    "communication_discipline": 0.10,
}

_TASK_PROFILE_WEIGHTS = {
    "ocr_text": {
        "target_semantics": 0.15,
        "evidence_acquisition": 0.40,
        "search_progress": 0.20,
        "tool_discipline": 0.15,
        "communication_discipline": 0.10,
    },
    "count_relation": {
        "target_semantics": 0.20,
        "evidence_acquisition": 0.35,
        "search_progress": 0.25,
        "tool_discipline": 0.10,
        "communication_discipline": 0.10,
    },
    "attribute": {
        "target_semantics": 0.25,
        "evidence_acquisition": 0.30,
        "search_progress": 0.20,
        "tool_discipline": 0.15,
        "communication_discipline": 0.10,
    },
    "general_search": _DEFAULT_COMPONENT_WEIGHTS,
}


def _clip_score(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        value = default
    return max(0.0, min(1.0, value))


def _default_process_meta() -> Dict[str, Any]:
    return {
        "process_score": 0.0,
        "task_profile": _DEFAULT_TASK_PROFILE,
        "components": {key: 0.0 for key in _COMPONENT_KEYS},
        "weights": dict(_DEFAULT_COMPONENT_WEIGHTS),
        "flags": [],
        "reasons": "",
    }


def _normalize_task_profile(value: Any) -> str:
    value = str(value or _DEFAULT_TASK_PROFILE)
    if value not in _TASK_PROFILE_WEIGHTS:
        return _DEFAULT_TASK_PROFILE
    return value


def _normalize_component_scores(raw_components: Any) -> Dict[str, float]:
    components = {}
    raw_components = raw_components if isinstance(raw_components, dict) else {}
    for key in _COMPONENT_KEYS:
        components[key] = _clip_score(raw_components.get(key, 0.0))
    return components


def _normalize_component_weights(raw_weights: Any, task_profile: str) -> Dict[str, float]:
    default_weights = _TASK_PROFILE_WEIGHTS.get(task_profile, _DEFAULT_COMPONENT_WEIGHTS)
    raw_weights = raw_weights if isinstance(raw_weights, dict) else {}

    weights: Dict[str, float] = {}
    total = 0.0
    for key in _COMPONENT_KEYS:
        try:
            value = float(raw_weights.get(key, default_weights[key]))
        except Exception:
            value = float(default_weights[key])
        value = max(0.0, value)
        weights[key] = value
        total += value

    if total <= 0:
        return dict(default_weights)

    return {key: weights[key] / total for key in _COMPONENT_KEYS}


def _build_process_meta_from_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return _default_process_meta()

    task_profile = _normalize_task_profile(obj.get("task_profile"))
    components = _normalize_component_scores(obj.get("components"))
    weights = _normalize_component_weights(obj.get("weights"), task_profile)
    process_score = sum(weights[key] * components[key] for key in _COMPONENT_KEYS)

    flags = obj.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(flag) for flag in flags if str(flag) in _FLAG_LIST]

    reasons = (obj.get("reasons", "") or "").replace("\n", " ").strip()

    return {
        "process_score": round(_clip_score(process_score), 4),
        "task_profile": task_profile,
        "components": {key: round(components[key], 4) for key in _COMPONENT_KEYS},
        "weights": {key: round(weights[key], 4) for key in _COMPONENT_KEYS},
        "flags": flags,
        "reasons": reasons,
    }


PROCESS_PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "v1_text": {
        "system": (
            "You are a strict process evaluator for multi-turn visual question answering with tool usage (crop/grounding).\n"
            "You will receive:\n"
            "A) an image,\n"
            "B) a human reference process written as concise Mini-o3-style round text,\n"
            "C) a model multi-turn rollout.\n\n"
            "Important:\n"
            "- The human reference process is already a distilled training template, not raw UI events.\n"
            "- Use it as a reference for what relevant regions, zoom granularity, and convergence pattern matter.\n"
            "- The model does NOT need to copy the reference wording exactly.\n"
            "- Alternative strategies are valid if they are equally evidence-driven, efficient, and convergent.\n"
            "- Score ONLY the model's process quality. Do NOT score final answer correctness.\n\n"
            "Process evaluation dimensions:\n"
            "1) Efficiency: minimal wasted steps and no repeated identical crops.\n"
            "2) Convergence: broad scan -> focused zoom -> verify/answer, or another clearly convergent pattern.\n"
            "3) Tool discipline: crops are purposeful and stop once evidence is sufficient.\n"
            "4) Relevance: model explores regions that match the question and are compatible with the human reference process.\n"
            "5) Communication discipline: avoids unnecessary self-talk.\n\n"
            "Scoring rubric:\n"
            "- 0.00~0.10: degenerate output, malformed answer structure, empty thinking, or immediate blind answer.\n"
            "- 0.10~0.30: no grounding/crop tool use for a visual search task, or near-pure guessing.\n"
            "- 0.30~0.50: minimal or ineffective tool use; random or irrelevant regions; no clear progress.\n"
            "- 0.50~0.70: partial exploration; some relevant crops but notable inefficiency or weak convergence.\n"
            "- 0.70~0.85: good exploration; mostly relevant, convergent, evidence-driven with minor inefficiencies.\n"
            "- 0.85~1.00: excellent; concise coarse-to-fine or otherwise efficient strategy, clear verification, prompt stop.\n\n"
            "Critical rules:\n"
            "- A model that uses NO grounding/crop tools for a visual search task MUST score <= 0.30.\n"
            "- A model that is much more complex than the human reference process should be penalized for inefficiency.\n"
            "- Do not reward long self-talk.\n"
            "- Do not assume shorter is always better; reward the minimum exploration needed for convincing evidence.\n"
            "- If the model explores different regions from the reference, that is acceptable only if the regions are still question-relevant and the process converges efficiently.\n\n"
            "Return ONLY JSON:\n"
            "{\"score\": <float 0..1>, \"reasons\": <short string>, \"flags\": <list of tags>}\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),
        "instruction": (
            "Use the human reference process text as a concise supervision signal for attention and search pattern.\n"
            "Reward relevant broad-to-fine exploration, purposeful crops, and clean convergence.\n"
            "Penalize random search, looping, irrelevant crops, premature answering, or excessive extra steps."
        ),
    },
    "v1_multidim_text": {
        "system": (
            "You are a process evaluator for multi-turn visual question answering with crop/grounding tool usage.\n"
            "You will receive:\n"
            "A) an image,\n"
            "B) a human reference process written as concise Mini-o3-style round text,\n"
            "C) a model multi-turn rollout.\n\n"

            "Important context for this v1 dataset:\n"
            "- The human reference process is text, not raw mouse events.\n"
            "- Use the reference as supervision for relevant attention, evidence regions, zoom granularity, and search pattern.\n"
            "- The model does NOT need to copy the reference wording or exact trajectory.\n"
            "- Alternative routes are valid if the evidence is sufficient and the search keeps progressing.\n"
            "- Score ONLY process quality. Final answer correctness is evaluated separately outside this judge.\n"
            "- Do NOT apply hard caps solely because the model used no crop/grounding; some answers may be visible from the original image.\n"
            "- If no crop is used, judge whether the original-image observation and reasoning provide sufficient evidence.\n"
            "- Do NOT reward a rollout merely because the final answer is correct; lucky or degraded correct rollouts should receive lower process scores.\n\n"

            "How to use the human reference process:\n"
            "- Use the human reference process to estimate task difficulty, ambiguity, and the likely evidence granularity needed for a trustworthy answer.\n"
            "- Human hesitation, revisits, zoom steps, and final_bboxes are meaningful supervision signals for what regions matter and how hard the task is.\n"
            "- Treat the human process as supervision for evidence needs, not as a required exact trajectory.\n"
            "- If the model can genuinely support the answer from broader viewing, it can still score well even when the human used finer inspection.\n\n"

            "First choose task_profile:\n"
            "- ocr_text: OCR, numbers, prices, time, phone numbers, signs, labels, or readable text.\n"
            "- count_relation: counting, multiple targets, left/right/up/down, relative position, or spatial relation.\n"
            "- attribute: person/object attribute recognition such as clothing, color, action, body type, gender, object state.\n"
            "- general_search: other visual search tasks.\n\n"

            "Task-adaptive weights:\n"
            f"{json.dumps(_TASK_PROFILE_WEIGHTS, ensure_ascii=False, sort_keys=True)}\n\n"

            "Score each component in [0,1]:\n"

            "1) target_semantics: Does the model understand the real target and queried attribute/relation/count?\n"
            "   Reward tracking the correct person/object/text/environment anchor and handling distractors.\n"

            "2) evidence_acquisition: Does the model obtain enough visual evidence for the answer?\n"
            "   For detail-heavy tasks, readable/specific crops are strong evidence. If the answer is genuinely visible without crop, no-crop can still score well.\n"

            "3) search_progress: Does each step add information?\n"
            "   Repeated language is acceptable when crops tighten, candidates are eliminated, branches change, or the process moves closer to final evidence.\n"
            "   Penalize only repetition without information gain.\n"

            "4) tool_discipline: If tools are used, are grounding JSON, bbox, source view, and crop granularity valid and useful?\n"
            "   Do not punish the absence of tool calls by itself; reflect missing visual evidence under evidence_acquisition/search_progress instead.\n"

            "5) communication_discipline: Is the reasoning concise and tied to visual uncertainty, comparison, or verification?\n"
            "   Penalize verbose self-talk, empty restatement, repeated claims that do not support an observation or next action, language drift, mixed-language confusion, emoji spam, repeated-character spam, or garbled text.\n\n"

            "Soft penalty rules:\n"
            "- If no crop is used but evidence is clearly sufficient, allow a high score and consider flag sufficient_no_crop_evidence.\n"
            "- If no crop is used and key text/number/small target/count/relation evidence is not actually visible, lower evidence_acquisition.\n"
            "- If crops drift away from the question target, reference process, or final_bboxes, lower target_semantics and evidence_acquisition.\n"
            "- If repeated crops provide no new evidence, lower search_progress and tool_discipline.\n"
            "- If a wrong branch is followed by a useful reset/refocus that reaches evidence, reward useful_reset/search_progress.\n"
            "- If the model keeps searching after evidence is sufficient, lower search_progress/tool_discipline but do not mechanically zero it.\n"
            "- If the rollout contains empty restatement, language drift, emotional self-talk, mixed-language fragments, excessive emojis, repeated characters, or garbled text, treat it as a degraded or collapsed process signal and assign a low communication_discipline score.\n"
            "- If such degeneration interferes with target tracking, evidence collection, search progress, or tool use, also lower the relevant components; even when the final answer is correct, give less process reward to collapsed or unstable rollouts.\n\n"

            "The final score must be the weighted sum of the five component scores using the selected task_profile weights, "
            "with only the soft penalties above reflected in the component scores. Do not use rigid score bands.\n\n"

            "Return ONLY JSON with this schema:\n"
            "{"
            "\"score\": <float 0..1>, "
            "\"task_profile\": \"ocr_text|count_relation|attribute|general_search\", "
            "\"components\": {"
            "\"target_semantics\": <float 0..1>, "
            "\"evidence_acquisition\": <float 0..1>, "
            "\"search_progress\": <float 0..1>, "
            "\"tool_discipline\": <float 0..1>, "
            "\"communication_discipline\": <float 0..1>"
            "}, "
            "\"weights\": {same keys as components, floats summing to 1}, "
            "\"reasons\": <short string>, "
            "\"flags\": <list of tags>"
            "}\n"
            f"Allowed flags: {_FLAG_LIST}\n"
        ),

        "instruction": (
            "Use the v1 human reference process text, image, final_bboxes, task_evaluation, and events_text_meta as supervision signals. "
            "Choose a task_profile, score the five process components, then return their weighted sum as score. "
            "Use the human process to judge expected difficulty, ambiguity, needed zoom granularity, and which local regions likely required careful inspection. "
            "Do not hard-cap no-crop answers; judge whether visual evidence is actually sufficient. "
            "Do not reward degraded, repetitive, language-drifting, garbled, or collapsed rollouts merely because the final answer is correct."
        ),
    },
}


def _normalize_reference_process(human_reference_process: Any) -> str:
    if human_reference_process is None:
        return ""
    if isinstance(human_reference_process, str):
        return human_reference_process
    try:
        return json.dumps(human_reference_process, ensure_ascii=False)
    except Exception:
        return str(human_reference_process)


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(value)


def _parse_process_json_response(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text

    text = "" if text is None else str(text).strip()
    if not text:
        raise ValueError("empty GPT process response")

    decoder = json.JSONDecoder()

    def _try_parse(candidate: Any) -> Dict[str, Any] | None:
        candidate = "" if candidate is None else str(candidate).strip()
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                obj, end = decoder.raw_decode(candidate[match.start():])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    parsed = _try_parse(text)
    if parsed is not None:
        return parsed

    fenced_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for candidate in fenced_matches:
        try:
            parsed = _try_parse(candidate)
            if parsed is not None:
                return parsed
        except Exception:
            continue

    raise ValueError(f"failed to parse GPT process JSON: {text[:300]}")


def _call_gpt_process_score(
    question_text: str,
    human_reference_process: Any,
    rollout_text: Any,
    sample_extra: Dict[str, Any],
) -> Dict[str, Any]:
    variant = str(GPT_RUNTIME.get("prompt_variant", "v1_multidim_text"))
    tmpl = PROCESS_PROMPT_TEMPLATES.get(variant, PROCESS_PROMPT_TEMPLATES["v1_multidim_text"])

    doc_id = sample_extra.get("doc_id", "")
    payload = {
        "task_type": "visual_qa_with_tool_calls",
        "question": question_text,
        "doc_id": doc_id,
        "annotation_time": sample_extra.get("annotation_time"),
        "task_evaluation": sample_extra.get("task_evaluation"),
        "final_bboxes": sample_extra.get("final_bboxes"),
        "events_text_meta": sample_extra.get("events_text_meta"),
        "human_reference_process": _normalize_reference_process(human_reference_process),
        "reference_process_type": "mini_o3_round_text",
        "model_rollout_round_text": rollout_text,
        "component_keys": _COMPONENT_KEYS,
        "default_component_weights": _DEFAULT_COMPONENT_WEIGHTS,
        "task_profile_weights": _TASK_PROFILE_WEIGHTS,
        "output_schema": {
            "score": "float[0,1]",
            "task_profile": "ocr_text|count_relation|attribute|general_search",
            "components": {key: "float[0,1]" for key in _COMPONENT_KEYS},
            "weights": {key: "float weight" for key in _COMPONENT_KEYS},
            "reasons": "string",
            "flags": f"subset of {_FLAG_LIST}",
        },
    }

    system_prompt = tmpl["system"]
    user_text = json.dumps(
        {"instruction": tmpl["instruction"], "payload": payload},
        ensure_ascii=False,
    )

    use_vision = int(GPT_RUNTIME.get("use_vision", 1)) == 1
    max_images = int(GPT_RUNTIME.get("max_images_to_gpt", 1))
    image_detail = GPT_RUNTIME.get("image_detail", "auto")
    log_score = int(GPT_RUNTIME.get("log_gpt_score", 1)) == 1

    img_data_urls: List[str] = []
    imgs = sample_extra.get("images")
    if imgs is None:
        img_items = []
    elif isinstance(imgs, str):
        img_items = [imgs]
    elif isinstance(imgs, (list, tuple)):
        img_items = list(imgs)
    elif hasattr(imgs, "tolist"):
        img_items = imgs.tolist()
        if isinstance(img_items, str):
            img_items = [img_items]
    else:
        img_items = [imgs]

    if use_vision and img_items:
        for rel_or_abs in img_items[:max_images]:
            abs_path = _resolve_image_abs(rel_or_abs)
            if abs_path:
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
            max_tokens=int(GPT_RUNTIME.get("process_max_tokens", 700)),
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

            obj = _parse_process_json_response(text)
            process_meta = _build_process_meta_from_obj(obj)
            score = float(process_meta["process_score"])

            if log_score:
                flags = process_meta["flags"]
                task_profile = process_meta["task_profile"]
                components = process_meta["components"]
                weights = process_meta["weights"]
                reasons = process_meta["reasons"]
                extra = ""
                if last_mode == "text_fallback" and vision_err is not None:
                    extra = f" vision_err={repr(vision_err)}"
                print(
                    f"[qa_process_v1][gpt_process] doc_id={doc_id} mode={last_mode} prompt={variant} "
                    f"score={score:.4f} task_profile={task_profile} "
                    f"components={_compact_json(components)} weights={_compact_json(weights)} "
                    f"reasons='{reasons}' flags={flags}{extra}"
                )

            return process_meta

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))

    if int(GPT_RUNTIME.get("log_gpt_score", 1)) == 1:
        print(f"[qa_process_v1][gpt_process] doc_id={doc_id} mode=failed prompt={variant} score=0.0000 err={repr(last_err)}")
    return _default_process_meta()


def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    extra_info = extra_info or {}

    acc_w = float(extra_info.get("acc_reward_weight", 1.0))
    fmt_w = float(extra_info.get("format_reward_weight", 0.0))
    proc_w = float(extra_info.get("process_reward_weight", 0.0))

    sample_extra = extra_info.get("sample_extra_info") or {}

    acc = _call_gpt_acc_score(question_text=prompt, predict_str_list=predict_str_list, ground_truth=ground_truth, extra_info=extra_info)
    acc_component = acc_w * float(acc)
    acc_is_correct = float(acc) >= 0.5

    fmt = _format_ok(predict_str_list)
    fmt_component = fmt_w * float(fmt)

    process_meta = _default_process_meta()
    process_eval_mode = "disabled" if proc_w <= 0 else "wrong_unstructured"
    if proc_w > 0 and acc_is_correct:
        process_eval_mode = "correct"
        process_meta = _call_gpt_process_score(
            question_text=prompt,
            human_reference_process=sample_extra.get("events"),
            rollout_text=predict_str_list,
            sample_extra=sample_extra,
        )

    process_score = float(process_meta["process_score"])
    proc_component = proc_w * float(process_score) * float(acc)
    format_component = fmt_component + proc_component
    total_mode = os.getenv("QA_PROCESS_TOTAL_MODE", "process").strip().lower()
    if total_mode in {"acc", "accuracy", "baseline", "answer"}:
        total = acc_component
    elif total_mode in {"acc_plus_process", "sum"}:
        total = acc_component + format_component
    else:
        total = format_component

    metrics = {
        "raw_accuracy": float(acc),
        "raw_format": float(fmt),
        "reward_total_mode": total_mode,
        "process_score": process_score,
        "process_target_semantics": float(process_meta["components"]["target_semantics"]),
        "process_evidence_acquisition": float(process_meta["components"]["evidence_acquisition"]),
        "process_search_progress": float(process_meta["components"]["search_progress"]),
        "process_tool_discipline": float(process_meta["components"]["tool_discipline"]),
        "process_communication_discipline": float(process_meta["components"]["communication_discipline"]),
        "process_task_profile": process_meta["task_profile"],
        "process_flags": process_meta["flags"],
        "process_reasons": process_meta["reasons"],
        "process_weights": process_meta["weights"],
        "process_eval_mode": process_eval_mode,
    }

    return {
        "score": float(total),
        "acc_score": float(acc_component),
        "format_score": float(format_component),
        "metrics": metrics,
    }
