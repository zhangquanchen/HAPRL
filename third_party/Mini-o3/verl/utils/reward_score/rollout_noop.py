"""No-op reward for rollout-only smoke tests.

This is intentionally separate from qa_process_v1. It lets val-only jobs
exercise the full multi-turn rollout path and save trajectories without
calling an external judge or contributing any reward signal.
"""


def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    return {
        "score": 0.0,
        "acc_score": 0.0,
        "format_score": 0.0,
        "metrics": {
            "raw_accuracy": 0.0,
            "raw_format": 0.0,
            "process_score": 0.0,
            "process_target_semantics": 0.0,
            "process_evidence_acquisition": 0.0,
            "process_search_progress": 0.0,
            "process_tool_discipline": 0.0,
            "process_communication_discipline": 0.0,
            "process_eval_mode": "rollout_noop",
        },
    }
