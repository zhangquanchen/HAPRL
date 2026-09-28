from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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
    _COMPONENT_KEYS,
    _DEFAULT_COMPONENT_WEIGHTS,
    _FLAG_LIST,
    _TASK_PROFILE_WEIGHTS,
    _build_process_meta_from_obj,
    _compact_json,
    _default_process_meta,
    _normalize_reference_process,
    _parse_process_json_response,
)
from .qa_process_v2 import PROCESS_PROMPT_TEMPLATES


_VALID_SOURCE_RE = re.compile(r"^observation_\d+$")
_PROCESS_ACC_THRESHOLD = 0.45
_LOW_PROCESS_MIN_ACC_SCALE = 0.70
_CHAR_REPEAT_THRESHOLD = 40
_CHUNK_REPEAT_THRESHOLD = 20
_OUTSIDE_ANSWER_OTHER_SCRIPT_THRESHOLD = 80


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _joined_rollout_text(predict_str_list: Any) -> str:
    if isinstance(predict_str_list, list):
        return "\n".join(str(x) for x in predict_str_list)
    return str(predict_str_list)


def _is_valid_bbox_2d(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        coords = [float(x) for x in value]
    except Exception:
        return False
    if any(x != x for x in coords):
        return False
    x1, y1, x2, y2 = coords
    return abs(x2 - x1) > 1e-6 and abs(y2 - y1) > 1e-6


def _is_valid_source(value: Any) -> bool:
    value = str(value or "")
    return value == "original_image" or bool(_VALID_SOURCE_RE.match(value))


def _parse_groundings(text: str) -> Tuple[List[Dict[str, Any]], int]:
    valid: List[Dict[str, Any]] = []
    invalid_count = 0
    blocks = re.findall(r"<grounding>\s*([\s\S]*?)\s*</grounding>", text, flags=re.IGNORECASE)
    for block in blocks:
        try:
            payload = json.loads(block.strip())
        except Exception:
            invalid_count += 1
            continue
        if not isinstance(payload, dict):
            invalid_count += 1
            continue
        bbox = payload.get("bbox_2d")
        source = payload.get("source")
        if not _is_valid_bbox_2d(bbox) or not _is_valid_source(source):
            invalid_count += 1
            continue
        valid.append({"bbox_2d": [float(x) for x in bbox], "source": str(source)})
    return valid, invalid_count


def _bbox_key(item: Dict[str, Any]) -> Tuple[str, Tuple[float, float, float, float]]:
    bbox = item["bbox_2d"]
    return item["source"], tuple(round(float(x), 3) for x in bbox)  # type: ignore[return-value]


def _duplicate_crop_rate(valid_groundings: List[Dict[str, Any]]) -> Tuple[int, float]:
    if len(valid_groundings) <= 1:
        return 0, 0.0
    seen = set()
    duplicates = 0
    for item in valid_groundings:
        key = _bbox_key(item)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates, duplicates / max(1, len(valid_groundings) - 1)


def _has_char_spam(text: str) -> bool:
    if re.search(rf"([^\s])\1{{{_CHAR_REPEAT_THRESHOLD - 1},}}", text):
        return True
    compact = re.sub(r"\s+", "", text)
    return bool(re.search(rf"(.{{2,8}})\1{{{_CHUNK_REPEAT_THRESHOLD - 1},}}", compact))


def _has_language_drift(text: str) -> bool:
    # Only catch large non-answer script drift. OCR answers may legitimately contain other scripts.
    other_script_chars = len(re.findall(r"[\u0400-\u052f\u0600-\u06ff\u0900-\u097f\u3040-\u30ff\uac00-\ud7af]", text))
    return other_script_chars >= _OUTSIDE_ANSWER_OTHER_SCRIPT_THRESHOLD


def _text_outside_answer_blocks(text: str) -> str:
    return re.sub(r"<answer>[\s\S]*?</answer>", " ", text, flags=re.IGNORECASE)


def _answer_tag_stats(text: str) -> Dict[str, Any]:
    open_matches = list(re.finditer(r"<answer>", text, flags=re.IGNORECASE))
    close_matches = list(re.finditer(r"</answer>", text, flags=re.IGNORECASE))
    pair_matches = list(re.finditer(r"<answer>\s*([\s\S]*?)\s*</answer>", text, flags=re.IGNORECASE))

    has_complete_answer = bool(pair_matches)
    repeated = len(pair_matches) > 1 or len(open_matches) > 1 or len(close_matches) > 1
    malformed = False
    post_garbage = False
    answer_text = ""

    if open_matches and close_matches:
        malformed = close_matches[0].start() <= open_matches[0].end()
    if len(open_matches) != len(close_matches):
        malformed = True
    if has_complete_answer:
        last_pair = pair_matches[-1]
        answer_text = last_pair.group(1).strip()
        post_garbage = bool(text[last_pair.end():].strip())

    return {
        "has_complete_answer": has_complete_answer,
        "answer_open_count": len(open_matches),
        "answer_close_count": len(close_matches),
        "answer_pair_count": len(pair_matches),
        "empty_answer": has_complete_answer and not answer_text,
        "repeated_answer_tags": repeated,
        "malformed_answer_tags": malformed,
        "post_answer_garbage": post_garbage,
    }


def _analyze_local_rollout(predict_str_list: Any) -> Dict[str, Any]:
    text = _joined_rollout_text(predict_str_list)
    fmt = float(_format_ok(predict_str_list))
    answer_stats = _answer_tag_stats(text)
    valid_groundings, invalid_grounding_count = _parse_groundings(text)
    valid_grounding_count = len(valid_groundings)
    duplicate_count, duplicate_rate = _duplicate_crop_rate(valid_groundings)
    tool_call_count = valid_grounding_count + invalid_grounding_count
    round_count = len(predict_str_list) if isinstance(predict_str_list, list) else 1
    response_char_length = len(text)

    text_outside_answer = _text_outside_answer_blocks(text)
    char_spam = _has_char_spam(text)
    language_drift = _has_language_drift(text_outside_answer)
    fatal_bad_reasons: List[str] = []
    soft_bad_reasons: List[str] = []
    if not answer_stats["has_complete_answer"] or fmt < 0.5:
        fatal_bad_reasons.append("missing_or_invalid_answer_tag")
    if answer_stats["malformed_answer_tags"]:
        fatal_bad_reasons.append("malformed_answer_tags")
    if answer_stats["empty_answer"]:
        fatal_bad_reasons.append("empty_answer")
    if answer_stats["repeated_answer_tags"]:
        soft_bad_reasons.append("repeated_answer_tags")
    if answer_stats["post_answer_garbage"]:
        soft_bad_reasons.append("post_answer_garbage")
    if invalid_grounding_count > 0:
        soft_bad_reasons.append("invalid_grounding")
    if char_spam:
        fatal_bad_reasons.append("char_spam")
    if language_drift:
        fatal_bad_reasons.append("language_drift")

    fatal_bad = bool(fatal_bad_reasons)
    soft_bad = bool(soft_bad_reasons) and not fatal_bad
    structured_rollout_ok = fmt >= 0.5 and valid_grounding_count > 0 and not fatal_bad and not soft_bad

    extra_tool_penalty = max(0, valid_grounding_count - 4) * 0.03
    duplicate_penalty = min(0.12, duplicate_rate * 0.12)
    length_penalty = min(0.12, max(0, response_char_length - 6000) / 10000)
    round_penalty = min(0.08, max(0, round_count - 5) * 0.04)
    local_penalty = min(0.30, extra_tool_penalty + duplicate_penalty + length_penalty + round_penalty)

    soft_bad_penalty = 0.0
    if answer_stats["repeated_answer_tags"]:
        soft_bad_penalty += 0.08
    if answer_stats["post_answer_garbage"]:
        soft_bad_penalty += 0.08
    if invalid_grounding_count > 0:
        soft_bad_penalty += min(0.12, invalid_grounding_count * 0.06)
    soft_bad_penalty = min(0.20, soft_bad_penalty)
    soft_bad_correct_cap = max(0.55, 0.75 - soft_bad_penalty)

    if fatal_bad:
        local_quality = 0.0
    elif structured_rollout_ok:
        local_quality = _clip01(1.0 - local_penalty / 0.30)
    elif soft_bad:
        local_quality = 0.0
    elif fmt >= 0.5:
        local_quality = 0.35
    else:
        local_quality = 0.0

    return {
        "raw_format": fmt,
        "structured_rollout_ok": float(structured_rollout_ok),
        "local_quality": float(local_quality),
        "local_penalty": float(local_penalty),
        "hard_bad": float(fatal_bad),
        "hard_bad_reason": ",".join(fatal_bad_reasons),
        "fatal_bad": float(fatal_bad),
        "fatal_bad_reason": ",".join(fatal_bad_reasons),
        "soft_bad": float(soft_bad),
        "soft_bad_reason": ",".join(soft_bad_reasons),
        "soft_bad_penalty": float(soft_bad_penalty),
        "soft_bad_correct_cap": float(soft_bad_correct_cap),
        "invalid_grounding_count": float(invalid_grounding_count),
        "valid_grounding_count": float(valid_grounding_count),
        "tool_call_count": float(tool_call_count),
        "round_count": float(round_count),
        "response_char_length": float(response_char_length),
        "duplicate_crop_count": float(duplicate_count),
        "duplicate_crop_rate": float(duplicate_rate),
        "answer_open_count": float(answer_stats["answer_open_count"]),
        "answer_close_count": float(answer_stats["answer_close_count"]),
        "answer_pair_count": float(answer_stats["answer_pair_count"]),
        "empty_answer": float(answer_stats["empty_answer"]),
        "repeated_answer_tags": float(answer_stats["repeated_answer_tags"]),
        "malformed_answer_tags": float(answer_stats["malformed_answer_tags"]),
        "post_answer_garbage": float(answer_stats["post_answer_garbage"]),
        "char_spam": float(char_spam),
        "language_drift": float(language_drift),
    }


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
    user_text = json.dumps({"instruction": tmpl["instruction"], "payload": payload}, ensure_ascii=False)

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
    last_err: Optional[Exception] = None
    vision_err: Optional[Exception] = None

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
                    f"[qa_process_v3][gpt_process] doc_id={doc_id} mode={last_mode} prompt={variant} "
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
        print(f"[qa_process_v3][gpt_process] doc_id={doc_id} mode=failed prompt={variant} score=0.0000 err={repr(last_err)}")
    return _default_process_meta()


def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    extra_info = extra_info or {}

    acc_w = float(extra_info.get("acc_reward_weight", 1.0))
    fmt_w = float(extra_info.get("format_reward_weight", 0.0))
    proc_w = float(extra_info.get("process_reward_weight", 0.0))
    sample_extra = extra_info.get("sample_extra_info") or {}

    local = _analyze_local_rollout(predict_str_list)
    fmt = float(local["raw_format"])
    fatal_bad = bool(local["fatal_bad"])
    hard_bad = fatal_bad
    soft_bad = bool(local["soft_bad"])
    structured_rollout_ok = bool(local["structured_rollout_ok"])

    acc = _call_gpt_acc_score(
        question_text=prompt,
        predict_str_list=predict_str_list,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    acc_component = acc_w * float(acc)
    acc_is_correct = float(acc) >= 0.5

    process_meta = _default_process_meta()
    process_score = 0.0
    proc_component = 0.0
    fmt_component = fmt_w * fmt
    wrong_struct_bonus = 0.0
    process_eval_mode = "wrong_unstructured"
    process_acc_discount = 1.0
    effective_acc_component = acc_component

    if fatal_bad:
        total = 0.0
        fmt_component = 0.0
        effective_acc_component = 0.0
        process_eval_mode = "fatal_bad"
    elif acc_is_correct and soft_bad:
        process_eval_mode = "correct_soft_bad"
        total = min(float(local["soft_bad_correct_cap"]), acc_component)
        effective_acc_component = total
        fmt_component = 0.0
    elif acc_is_correct:
        process_eval_mode = "correct"
        if proc_w > 0:
            process_meta = _call_gpt_process_score(
                question_text=prompt,
                human_reference_process=sample_extra.get("events"),
                rollout_text=predict_str_list,
                sample_extra=sample_extra,
            )
            process_score = float(process_meta["process_score"])
            proc_component = proc_w * process_score
            if process_score < _PROCESS_ACC_THRESHOLD:
                process_acc_discount = _LOW_PROCESS_MIN_ACC_SCALE + (
                    (1.0 - _LOW_PROCESS_MIN_ACC_SCALE) * _clip01(process_score / _PROCESS_ACC_THRESHOLD)
                )
                effective_acc_component = acc_component * process_acc_discount
        process_format_component = max(0.0, fmt_component + proc_component - float(local["local_penalty"]))
        total = effective_acc_component + process_format_component
        fmt_component = process_format_component
    elif structured_rollout_ok:
        process_eval_mode = "wrong_structured"
        wrong_struct_bonus = proc_w * 0.10 * float(local["local_quality"])
        fmt_component = fmt_w * fmt + wrong_struct_bonus
        total = fmt_component
        effective_acc_component = 0.0
    elif soft_bad:
        process_eval_mode = "wrong_soft_bad"
        total = 0.0
        fmt_component = 0.0
        effective_acc_component = 0.0
    else:
        total = 0.0
        fmt_component = 0.0
        effective_acc_component = 0.0

    effective_fmt = fmt if (acc_is_correct or structured_rollout_ok) and not fatal_bad else 0.0

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
        "process_eval_mode": process_eval_mode,
        "process_acc_discount": float(process_acc_discount),
        "structured_rollout_ok": float(structured_rollout_ok),
        "local_quality": float(local["local_quality"]),
        "local_penalty": float(local["local_penalty"]),
        "hard_bad": float(hard_bad),
        "hard_bad_reason": local["hard_bad_reason"],
        "fatal_bad": float(fatal_bad),
        "fatal_bad_reason": local["fatal_bad_reason"],
        "soft_bad": float(soft_bad),
        "soft_bad_reason": local["soft_bad_reason"],
        "soft_bad_penalty": float(local["soft_bad_penalty"]),
        "soft_bad_correct_cap": float(local["soft_bad_correct_cap"]),
        "invalid_grounding_count": float(local["invalid_grounding_count"]),
        "valid_grounding_count": float(local["valid_grounding_count"]),
        "tool_call_count": float(local["tool_call_count"]),
        "round_count": float(local["round_count"]),
        "response_char_length": float(local["response_char_length"]),
        "duplicate_crop_count": float(local["duplicate_crop_count"]),
        "duplicate_crop_rate": float(local["duplicate_crop_rate"]),
        "answer_open_count": float(local["answer_open_count"]),
        "answer_close_count": float(local["answer_close_count"]),
        "answer_pair_count": float(local["answer_pair_count"]),
        "empty_answer": float(local["empty_answer"]),
        "repeated_answer_tags": float(local["repeated_answer_tags"]),
        "malformed_answer_tags": float(local["malformed_answer_tags"]),
        "post_answer_garbage": float(local["post_answer_garbage"]),
        "char_spam": float(local["char_spam"]),
        "language_drift": float(local["language_drift"]),
        "wrong_struct_bonus_applied": float(wrong_struct_bonus > 0.0),
        "wrong_struct_bonus": float(wrong_struct_bonus),
    }

    return {
        "score": float(max(0.0, total)),
        "acc_score": float(max(0.0, effective_acc_component)),
        "format_score": float(fmt_component),
        "metrics": metrics,
    }
