from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from .qa_process import (
    OPENAI_GATEWAY,
    GPT_RUNTIME,
    _call_gpt_acc_score,
    _file_to_data_url_cached,
    _format_ok,
    _get_client,
    _process_gate,
    _resolve_image_abs,
)
from .qa_process_v1 import (
    _FLAG_LIST,
    _COMPONENT_KEYS,
    _DEFAULT_COMPONENT_WEIGHTS,
    _TASK_PROFILE_WEIGHTS,
    _build_process_meta_from_obj,
    _compact_json,
    _default_process_meta,
    _normalize_reference_process,
    _parse_process_json_response,
)


_VALID_SOURCE_RE = re.compile(r"^observation_\d+$")

_V1_MULTIDIM_PROCESS_TEMPLATE = {
    "system": (
        "You are a process evaluator for multi-turn visual question answering with crop/grounding tool usage.\n"
        "You will receive:\n"
        "A) an image,\n"
        "B) a human reference process written as concise Mini-o3-style round text,\n"
        "C) a model multi-turn rollout.\n\n"
        "Important context for this process-text QA dataset:\n"
        "- The human reference process is text, not raw mouse events.\n"
        "- Use the reference as supervision for relevant attention, evidence regions, zoom granularity, and search pattern.\n"
        "- The model does NOT need to copy the reference wording or exact trajectory.\n"
        "- Alternative routes are valid if the evidence is sufficient and the search keeps progressing.\n"
        "- Score ONLY process quality. Final answer correctness is evaluated separately outside this judge.\n"
        "- Downstream training may use this process score for both correct samples and a very small structured subset of wrong samples. Keep your judgment purely about process quality, not answer correctness.\n"
        "- Do NOT apply hard caps solely because the model used no crop/grounding; some answers may be visible from the original image.\n"
        "- If no crop is used, judge whether the original-image observation and reasoning provide sufficient evidence.\n"
        "- Do NOT default the five component scores to similar values. Identify the dominant weakness first, then lower the most relevant component(s).\n\n"
        "How to use the human reference process:\n"
        "- The human process reveals where the task is difficult, ambiguous, or distractor-heavy.\n"
        "- Pay attention to the human's hesitation, revisits, zoom granularity, and final bbox regions as clues for what evidence was actually necessary.\n"
        "- Treat the human process as a clue about likely difficulty and useful evidence regions, not as a requirement that the model must inspect exactly the same way.\n"
        "- If the model can genuinely support the answer from broader viewing, that can still score well even when the human used finer local inspection.\n"
        "- If the human process is simple and direct, reward model trajectories that are similarly economical; if the human process is complex, do not over-penalize necessary extra inspection.\n\n"
        "First choose task_profile:\n"
        "- ocr_text: OCR, numbers, prices, time, phone numbers, signs, labels, or readable text.\n"
        "- count_relation: counting, multiple targets, left/right/up/down, relative position, or spatial relation.\n"
        "- attribute: person/object attribute recognition such as clothing, color, action, body type, gender, object state.\n"
        "- general_search: other visual search tasks.\n\n"
        "Task-adaptive weights:\n"
        f"{json.dumps(_TASK_PROFILE_WEIGHTS, ensure_ascii=False, sort_keys=True)}\n\n"
        "Score each component in [0,1]. Use these anchors:\n"
        "1) target_semantics: Does the model stay focused on the correct target, attribute, relation, count, or text anchor?\n"
        "   0.85~1.00: quickly locks onto the right target and stays there.\n"
        "   0.60~0.80: brief detour, then a useful reset returns to the right target.\n"
        "   0.30~0.50: repeated target confusion or unstable interpretation of the queried relation.\n"
        "   0.00~0.20: most of the process follows the wrong target or wrong region.\n"
        "   Lower this when left/right/up/down/under/behind or the target identity is repeatedly reinterpreted.\n"
        "2) evidence_acquisition: Does the model obtain enough visual evidence for the final answer?\n"
        "   0.85~1.00: final evidence is clear and directly supports the answer.\n"
        "   0.60~0.80: evidence is mostly sufficient but crops are slightly broad or weakly verified.\n"
        "   0.30~0.50: relevant region is found but key detail remains weak or unverified.\n"
        "   0.00~0.20: answer appears lucky; the process never secures convincing evidence.\n"
        "   Lower this for broad-only viewing, weak final crops, unsupported deduction after failed crops, clock-convention guessing, premise-arguing, or when key text/color/count/relation detail is not actually visible.\n"
        "3) search_progress: Does each step reduce uncertainty?\n"
        "   0.85~1.00: clear coarse-to-fine or another strongly convergent pattern; each step adds new information.\n"
        "   0.60~0.80: mostly progressive with only minor redundant revisits.\n"
        "   0.30~0.50: some progress, but mixed with noticeable no-gain loops.\n"
        "   0.00~0.20: repeated re-checks, repeated returns to the same observation or original image, repeated self-corrections without new evidence, or long sections that mainly rewrite failed history.\n"
        "   Repetition is acceptable only if crops tighten, candidates are eliminated, or a branch meaningfully changes.\n"
        "4) tool_discipline: If tools are used, are grounding JSON, bbox, source view, and crop granularity valid and useful?\n"
        "   0.85~1.00: tools are used sparingly and precisely; bbox/source/granularity are well controlled.\n"
        "   0.60~0.80: a small amount of trial-and-error, but overall useful.\n"
        "   0.30~0.50: repeated wide crops, repeated crops of the same area, or unstable source usage.\n"
        "   0.00~0.20: tool misuse, wrong source view, invalid JSON, or many ineffective crops.\n"
        "   Do not punish the absence of tool calls by itself; reflect missing visual evidence under evidence_acquisition/search_progress instead.\n"
        "5) communication_discipline: Is the reasoning concise, stable, and tied to visual uncertainty, comparison, or verification?\n"
        "   0.85~1.00: concise and evidence-oriented.\n"
        "   0.60~0.80: slightly verbose but still grounded in evidence and next actions.\n"
        "   0.30~0.50: noticeably long or repetitive with limited informational value.\n"
        "   0.00~0.15: mixed-language drift, gibberish or abnormal strings, emotional self-dialogue, repeated short-phrase spam, repeated claims without visual gain, or literary expansion detached from evidence.\n\n"
        "Decision principles:\n"
        "- First identify the dominant weakness of the rollout, then lower the most relevant component(s).\n"
        "- Do NOT mechanically lower all five components because one dimension is weak.\n"
        "- Do NOT give a mid/high process score just because the final answer is correct.\n"
        "- Do NOT automatically penalize long processes; penalize long processes only when they add little or no new evidence.\n"
        "- Do NOT automatically penalize many tool calls; penalize many tool calls only when they are repeated, broad, drifting, invalid, or no-gain.\n"
        "- Use the human reference process to estimate task difficulty, ambiguity, and the likely evidence granularity needed for a trustworthy answer.\n"
        "- Human hesitation, revisits, zoom steps, and final_bboxes are meaningful supervision signals for what regions matter and how hard the task is.\n"
        "- If no crop is used but evidence is clearly sufficient, allow a high score and consider flag sufficient_no_crop_evidence.\n"
        "- If a wrong branch is followed by a useful reset that reaches evidence, reflect that by keeping search_progress or target_semantics from collapsing.\n"
        "- If the model keeps searching after evidence is sufficient, lower search_progress or tool_discipline.\n"
        "- If the rollout becomes repetitive, mixed-language, emotionally self-referential, or gibberish-like without adding evidence, reflect that primarily in communication_discipline.\n"
        "- If the rollout repeatedly revisits the same observation, repeatedly says wait/reconsider/from scratch, or mainly rewrites failed reasoning without new crops, reflect that primarily in search_progress.\n"
        "- If the rollout replaces visual confirmation with premise arguments, convention-based guessing, or unsupported inference after failed crops, reflect that primarily in evidence_acquisition.\n\n"
        "The final score must be the weighted sum of the five component scores using the selected task_profile weights.\n\n"
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
        "Use the human reference process text, image, final_bboxes, task_evaluation, and events_text_meta as supervision signals. "
        "Choose a task_profile, identify the dominant strengths and weaknesses of the rollout, score the five components separately, "
        "and return their weighted sum as score. Use the human process to judge expected difficulty, needed zoom granularity, and which local regions likely required careful inspection. "
        "Do not hard-cap no-crop answers; judge whether visual evidence is actually sufficient. "
        "When the rollout shows mixed-language drift, gibberish, emotional self-dialogue, repeated observation fixation, unsupported deduction, or long no-gain search, lower the corresponding component sharply rather than flattening every component."
    ),
}

PROCESS_PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "v1_multidim_text": _V1_MULTIDIM_PROCESS_TEMPLATE,
}


def _joined_rollout_text(predict_str_list: Any) -> str:
    if isinstance(predict_str_list, list):
        return "\n".join(str(x) for x in predict_str_list)
    return str(predict_str_list)


def _is_valid_bbox_2d(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        [float(x) for x in value]
    except Exception:
        return False
    return True


def _is_valid_source(value: Any) -> bool:
    value = str(value or "")
    return value == "original_image" or bool(_VALID_SOURCE_RE.match(value))


def _structured_rollout_ok(predict_str_list: Any) -> bool:
    if _format_ok(predict_str_list) < 0.5:
        return False

    text = _joined_rollout_text(predict_str_list)
    grounding_blocks = re.findall(r"<grounding>\s*([\s\S]*?)\s*</grounding>", text, flags=re.IGNORECASE)
    if not grounding_blocks:
        return False

    for block in grounding_blocks:
        try:
            payload = json.loads(block.strip())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not _is_valid_bbox_2d(payload.get("bbox_2d")):
            continue
        if not _is_valid_source(payload.get("source")):
            continue
        return True

    return False


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
    if use_vision and isinstance(imgs, list) and imgs:
        for rel_or_abs in imgs[:max_images]:
            abs_path = _resolve_image_abs(rel_or_abs)
            if abs_path:
                data_url = _file_to_data_url_cached(abs_path)
                if data_url:
                    img_data_urls.append(data_url)

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
            for data_url in img_data_urls:
                content.append({"type": "image_url", "image_url": {"url": data_url, "detail": image_detail}})
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
                except Exception as err:
                    vision_err = err
                    last_mode = "text_fallback"
                    text = _run(with_images=False)

            obj = _parse_process_json_response(text)
            process_meta = _build_process_meta_from_obj(obj)
            score = float(process_meta["process_score"])

            if log_score:
                extra = ""
                if last_mode == "text_fallback" and vision_err is not None:
                    extra = f" vision_err={repr(vision_err)}"
                print(
                    f"[qa_process_v2][gpt_process] doc_id={doc_id} mode={last_mode} prompt={variant} "
                    f"score={score:.4f} task_profile={process_meta['task_profile']} "
                    f"components={_compact_json(process_meta['components'])} "
                    f"weights={_compact_json(process_meta['weights'])} "
                    f"reasons='{process_meta['reasons']}' flags={process_meta['flags']}{extra}"
                )

            return process_meta
        except Exception as err:
            last_err = err
            if attempt == max_retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))

    if int(GPT_RUNTIME.get("log_gpt_score", 1)) == 1:
        print(f"[qa_process_v2][gpt_process] doc_id={doc_id} mode=failed prompt={variant} score=0.0000 err={repr(last_err)}")
    return _default_process_meta()


def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    extra_info = extra_info or {}

    acc_w = float(extra_info.get("acc_reward_weight", 1.0))
    fmt_w = float(extra_info.get("format_reward_weight", 0.0))
    proc_w = float(extra_info.get("process_reward_weight", 0.0))

    sample_extra = extra_info.get("sample_extra_info") or {}

    acc = _call_gpt_acc_score(
        question_text=prompt,
        predict_str_list=predict_str_list,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    acc_component = acc_w * float(acc)
    acc_is_correct = float(acc) >= 0.5

    fmt = _format_ok(predict_str_list)
    process_meta = _default_process_meta()
    structured_rollout_ok = _structured_rollout_ok(predict_str_list)
    effective_fmt = float(fmt)
    if not acc_is_correct:
        effective_fmt = float(structured_rollout_ok)
    fmt_component = fmt_w * effective_fmt
    wrong_process_score = 0.0
    wrong_process_bonus_applied = 0.0
    process_eval_mode = "wrong_unstructured"

    if proc_w > 0 and acc_is_correct:
        process_eval_mode = "correct"
        process_meta = _call_gpt_process_score(
            question_text=prompt,
            human_reference_process=sample_extra.get("events"),
            rollout_text=predict_str_list,
            sample_extra=sample_extra,
        )
        process_score = float(process_meta["process_score"])
        proc_component = proc_w * process_score
        format_component = fmt_component + proc_component
        #total = acc_component + format_component
        total = format_component
    elif structured_rollout_ok:
        process_eval_mode = "wrong_structured"
        if proc_w > 0:
            process_meta = _call_gpt_process_score(
                question_text=prompt,
                human_reference_process=sample_extra.get("events"),
                rollout_text=predict_str_list,
                sample_extra=sample_extra,
            )
            wrong_process_score = float(process_meta["process_score"])
            if wrong_process_score > 0.7:
                wrong_process_bonus_applied = 1.0
        wrong_proc_component = 0.10 * proc_w if wrong_process_bonus_applied > 0 else 0.0
        format_component = fmt_component + wrong_proc_component
        total = format_component
        process_score = wrong_process_score
    else:
        process_score = 0.0
        format_component = 0.0
        total = 0.0
        effective_fmt = 0.0

    metrics = {
        "raw_accuracy": float(acc),
        "raw_format": float(effective_fmt),
        "process_score": float(process_score),
        "process_target_semantics": float(process_meta["components"]["target_semantics"]),
        "process_evidence_acquisition": float(process_meta["components"]["evidence_acquisition"]),
        "process_search_progress": float(process_meta["components"]["search_progress"]),
        "process_tool_discipline": float(process_meta["components"]["tool_discipline"]),
        "process_communication_discipline": float(process_meta["components"]["communication_discipline"]),
        "process_task_profile": process_meta["task_profile"],
        "process_flags": process_meta["flags"],
        "process_reasons": process_meta["reasons"],
        "process_weights": process_meta["weights"],
        "structured_rollout_ok": float(structured_rollout_ok),
        "wrong_process_score": float(wrong_process_score),
        "wrong_process_bonus_applied": float(wrong_process_bonus_applied),
        "process_eval_mode": process_eval_mode,
    }

    return {
        "score": float(total),
        "acc_score": float(acc_component),
        "format_score": float(format_component),
        "metrics": metrics,
    }
