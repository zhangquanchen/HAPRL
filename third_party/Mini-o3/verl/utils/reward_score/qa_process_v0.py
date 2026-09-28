# verl/utils/reward_score/qa_process_v0.py
"""
QA reward: same data and process judge (prompts, v1_text) as qa_process_v1, but
reward aggregation matches qa_process.py (process only when correct and proc_w>0;
proc_component = process_score * acc * proc_w; no process_reward gate).
"""

from __future__ import annotations

from .qa_process import _call_gpt_acc_score, _format_ok
from .qa_process_v1 import _call_gpt_process_score


def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    """
    Returns (total, acc_component, format_component).
    - acc_component: acc_w * acc
    - format_component: fmt_w * format_ok + process_score * acc * proc_w
    - Process GPT: only when proc_w > 0 and accuracy is correct (same as qa_process).
    """
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
    fmt_component = fmt_w * float(fmt)

    acc_val = float(acc)
    if proc_w > 0 and acc_is_correct:
        process_score = _call_gpt_process_score(
            question_text=prompt,
            human_reference_process=sample_extra.get("events"),
            rollout_text=predict_str_list,
            sample_extra=sample_extra,
        )
    else:
        process_score = 0.0

    proc_component = float(process_score) * acc_val * proc_w
    format_component = fmt_component + proc_component
    total = acc_component + format_component
    #total = format_component
    return total, acc_component, format_component
