# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
# from verl.utils.reward_score.openr1 import format_reward, acc_reward
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
# import logging
# import json


_NUMERIC_RESULT_METRIC_MAP = {
    "raw_accuracy": "raw_acc_scores",
    "raw_format": "raw_format_scores",
    "process_score": "process_scores",
    "process_target_semantics": "process_target_semantics",
    "process_evidence_acquisition": "process_evidence_acquisition",
    "process_search_progress": "process_search_progress",
    "process_tool_discipline": "process_tool_discipline",
    "process_communication_discipline": "process_communication_discipline",
    "process_acc_discount": "process_acc_discount",
    "structured_rollout_ok": "structured_rollout_ok",
    "wrong_process_score": "wrong_process_score",
    "wrong_process_bonus_applied": "wrong_process_bonus_applied",
    "local_quality": "local_quality",
    "local_penalty": "local_penalty",
    "hard_bad": "hard_bad",
    "fatal_bad": "fatal_bad",
    "soft_bad": "soft_bad",
    "soft_bad_penalty": "soft_bad_penalty",
    "soft_bad_correct_cap": "soft_bad_correct_cap",
    "invalid_grounding_count": "invalid_grounding_count",
    "valid_grounding_count": "valid_grounding_count",
    "tool_call_count": "tool_call_count",
    "round_count": "round_count",
    "response_char_length": "response_char_length",
    "duplicate_crop_count": "duplicate_crop_count",
    "duplicate_crop_rate": "duplicate_crop_rate",
    "answer_open_count": "answer_open_count",
    "answer_close_count": "answer_close_count",
    "answer_pair_count": "answer_pair_count",
    "empty_answer": "empty_answer",
    "repeated_answer_tags": "repeated_answer_tags",
    "malformed_answer_tags": "malformed_answer_tags",
    "post_answer_garbage": "post_answer_garbage",
    "char_spam": "char_spam",
    "language_drift": "language_drift",
    "wrong_struct_bonus_applied": "wrong_struct_bonus_applied",
    "wrong_struct_bonus": "wrong_struct_bonus",
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class NaiveMultiThreadsToolRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

        self.extra_info = kwargs.get("extra_info", {})
        self.gpt_threads = kwargs.get("gpt_threads", 100)

        self.overlong_buffer_len = self.extra_info.get("overlong_buffer_len", 0)
        self.max_total_response_length = self.extra_info['max_total_response_length']

    def extract_responses_list(
        self, 
        tokenizer, 
        input_ids: torch.Tensor, # 4096 + 8192
        multi_turn_response_mask: torch.Tensor # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        # Tensor Method
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0]
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s:e+1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        
        return decoded_responses

    #---------------------------------
    @staticmethod
    def _normalize_images(imgs):
        """
        Normalize various image container types into python list[str] or None.
        Accept:
          - None
          - str
          - list[str]
          - tuple[str]
          - numpy.ndarray (dtype=object or string)
        Return:
          - list[str] or None
        """
        if imgs is None:
            return None

        # single path
        if isinstance(imgs, str):
            return [imgs]

        # list/tuple
        if isinstance(imgs, (list, tuple)):
            out = [x for x in imgs if isinstance(x, str) and len(x) > 0]
            return out if len(out) > 0 else None

        # numpy array
        try:
            import numpy as np
            if isinstance(imgs, np.ndarray):
                flat = imgs.reshape(-1).tolist()
                out = [x for x in flat if isinstance(x, str) and len(x) > 0]
                return out if len(out) > 0 else None
        except Exception:
            pass

        return None

    #--------------------------------
    def process_single(self, args):

        index, data_item = args

        # safe debug (no undefined vars)
        try:
            _keys = sorted(list(getattr(data_item, "non_tensor_batch", {}).keys()))
            print("[debug][reward] non_tensor keys:", _keys[:50])
            print("[debug][reward] doc_id:", repr(getattr(data_item, "non_tensor_batch", {}).get("doc_id", None)))
            print("[debug][reward] images:", repr(getattr(data_item, "non_tensor_batch", {}).get("images", None)))
        except Exception as e:
            print("[debug][reward] failed:", repr(e))
        
        time_start = time.time()

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        # For multi turn, we maybe need `response_str` in a list format
        if 'multi_turn_response_mask' in data_item.batch:
            # `response_str` is a list now
            response_str = self.extract_responses_list(
                self.tokenizer,
                data_item.batch['input_ids'],
                data_item.batch['multi_turn_response_mask']
            )
        else:
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = [response_str]

        ground_truth = data_item.non_tensor_batch['ground_truth']

        data_source = data_item.non_tensor_batch['data_source']

        #extra_info = self.extra_info

        #------
        # NOTE: `self.extra_info` is global (from hydra).
        # For iou_process we need per-sample metadata (events/annotation_time/task_evaluation)
        # and also image relative paths (images) so reward can resolve images via BASE_IMAGE_DIR.
        extra_info = dict(self.extra_info) if self.extra_info is not None else {}
        # -----------------------------
        # 1) Per-sample extra_info dict (events, annotation_time, task_evaluation, ...)
        # -----------------------------
        sample_extra = {}
        ex = data_item.non_tensor_batch.get("extra_info", None)
        if isinstance(ex, dict):
            sample_extra = dict(ex)  # shallow copy, safe
        # -----------------------------
        # 2) Inject per-sample images (relative paths) for GPT vision or logging
        # -----------------------------
        #imgs = data_item.non_tensor_batch.get("images", None)
        #if isinstance(imgs, list) and len(imgs) > 0:
            #sample_extra["images"] = imgs
        imgs_raw = data_item.non_tensor_batch.get("images", None)
        imgs = self._normalize_images(imgs_raw)
        if imgs is not None:
            sample_extra["images"] = imgs

        # -----------------------------
        # 3) Inject doc_id (requires ray_trainer.py to include 'doc_id' in non_tensor_batch_keys)
        # -----------------------------

        doc_id = data_item.non_tensor_batch.get("doc_id", None)
        if doc_id is not None:
            sample_extra["doc_id"] = str(doc_id)


        extra_info["sample_extra_info"] = sample_extra
        #------
        # ✅ 下面这句保留，不要删
        question = data_item.non_tensor_batch['raw_prompt']

        #------------------------------
        """
        # ===================== DEBUG: what will be sent to compute_score =====================
        try:
            nt = data_item.non_tensor_batch

            _doc_id = nt.get("doc_id", None)
            _imgs = nt.get("images", None)
            _has_mm = ("multi_modal_data" in nt)
            _imgs_norm = self._normalize_images(_imgs)
            print(f"[debug][reward] doc_id={doc_id!r} has_images={imgs is not None} "
                  f"raw_images_type={type(imgs_raw)} norm_images_head={(imgs[:1] if imgs else None)}")


            # show first image path & existence after BASE_IMAGE_DIR join (critical for mode=vision)
            if isinstance(_imgs, list) and len(_imgs) > 0:
                import os
                base = os.environ.get("BASE_IMAGE_DIR", "")
                joined = os.path.normpath(os.path.join(base, _imgs[0]))
                print(f"[debug][reward] BASE_IMAGE_DIR={base} first_rel={_imgs[0]} joined={joined} exists={os.path.exists(joined)}")

            # show what we actually injected into extra_info
            _sei = extra_info.get("sample_extra_info", {})
            print(f"[debug][reward] injected sample_extra_info keys={list(_sei.keys())} "
                f"doc_id_injected={repr(_sei.get('doc_id', None))} "
                f"images_injected_type={type(_sei.get('images', None))}")

            # prompt sanity (avoid printing huge prompt)
            _q = nt.get("raw_prompt", None)
            if isinstance(_q, str):
                print(f"[debug][reward] raw_prompt_head={repr(_q[:120])}")
        except Exception as e:
            print(f"[debug][reward] debug_failed: {repr(e)}")
        # =================== END DEBUG ===================
        """
        #---------------------------------
        result = self.compute_score(
            prompt=question,
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        if isinstance(result, dict) and result.get("is_filter", False):
            return (index, result)

        metrics = {}
        if isinstance(result, dict):
            score = _safe_float(result.get("score", 0.0))
            acc_score = _safe_float(result.get("acc_score", 0.0))
            format_score = _safe_float(result.get("format_score", 0.0))
            metrics = result.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
        elif isinstance(result, (tuple, list)) and len(result) >= 3:
            score = _safe_float(result[0], 0.0)
            acc_score = _safe_float(result[1], 0.0)
            format_score = _safe_float(result[2], 0.0)
        else:
            score = _safe_float(result, 0.0)
            acc_score = 0.0
            format_score = 0.0

        overlong_buffer_len = self.overlong_buffer_len
        if overlong_buffer_len > 0:
            expected_len = self.max_total_response_length - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = 1.0
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            score += overlong_reward
        else:
            overlong_reward = 0

        time_end = time.time()

        return (
            index,
            score,
            acc_score,
            format_score,
            overlong_reward,
            valid_response_length,
            data_source,
            prompt_str,
            response_str,
            ground_truth,
            metrics,
            time_end - time_start,
        )

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_reward_tensor = reward_tensor.clone()
        format_reward_tensor = reward_tensor.clone()
        overlong_reward_tensor = reward_tensor.clone()
        sample_tensor_metrics = {}
        sample_non_tensor_metrics = {}

        already_print_data_sources = {}

        time_start = time.time()

        n_threads = self.gpt_threads

        print("gpt_threads: ", n_threads)

        # import pdb; pdb.set_trace()

        executor = ThreadPoolExecutor(n_threads)
        futures = []
        for i in range(len(data)):
            futures.append(executor.submit(self.process_single, (i, data[i])))
        results = [f.result() for f in as_completed(futures)]

        results = sorted(results, key=lambda x: x[0])
        results = [result[1:] for result in results]

        time_consume_list = []
        invalid_uids = []

        def ensure_sample_tensor_metric(key: str):
            if key not in sample_tensor_metrics:
                sample_tensor_metrics[key] = torch.zeros((len(data),), dtype=torch.float32)
            return sample_tensor_metrics[key]

        def ensure_sample_non_tensor_metric(key: str, default_value=""):
            if key not in sample_non_tensor_metrics:
                sample_non_tensor_metrics[key] = np.full((len(data),), default_value, dtype=object)
            return sample_non_tensor_metrics[key]

        for i, result in enumerate(results):

            if isinstance(result[0], dict) and result[0]['is_filter'] == True:
                invalid_uids.append(data[i].non_tensor_batch['uid'])
                continue
            
            (
                score,
                acc_score,
                format_score,
                overlong_score,
                valid_response_length,
                data_source,
                prompt_str,
                response_str,
                ground_truth,
                metrics,
                time_consume,
            ) = result
            reward_tensor[i, valid_response_length - 1] = score
            acc_reward_tensor[i, valid_response_length - 1] = acc_score
            format_reward_tensor[i, valid_response_length - 1] = format_score
            overlong_reward_tensor[i, valid_response_length - 1] = overlong_score

            if isinstance(metrics, dict):
                for src_key, dst_key in _NUMERIC_RESULT_METRIC_MAP.items():
                    if src_key in metrics:
                        ensure_sample_tensor_metric(dst_key)[i] = _safe_float(metrics.get(src_key, 0.0))
                if "process_task_profile" in metrics:
                    ensure_sample_non_tensor_metric("process_task_profile")[i] = str(metrics.get("process_task_profile") or "")
                if "process_eval_mode" in metrics:
                    ensure_sample_non_tensor_metric("process_eval_mode")[i] = str(metrics.get("process_eval_mode") or "")
                if "hard_bad_reason" in metrics:
                    ensure_sample_non_tensor_metric("hard_bad_reason")[i] = str(metrics.get("hard_bad_reason") or "")
                if "fatal_bad_reason" in metrics:
                    ensure_sample_non_tensor_metric("fatal_bad_reason")[i] = str(metrics.get("fatal_bad_reason") or "")
                if "soft_bad_reason" in metrics:
                    ensure_sample_non_tensor_metric("soft_bad_reason")[i] = str(metrics.get("soft_bad_reason") or "")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < 5:
                already_print_data_sources[data_source] += 1
                if isinstance(response_str, list):
                    response_str = ' '.join(response_str)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", (score, acc_score, format_score))
                print("[overlong_reward]", (overlong_score))
            
            time_consume_list.append(time_consume)

        time_consume_list = sorted(time_consume_list, reverse=True)
        print("top 10 time consuming in reward fn: ", time_consume_list[:10])

        print(f"there are {len(invalid_uids)} invalid samples in this batch: {invalid_uids[:5]}")

        time_end = time.time()

        print("total time: ", time_end - time_start)

        sample_metrics = None
        if sample_tensor_metrics or sample_non_tensor_metrics:
            sample_metrics = {
                "tensor_metrics": sample_tensor_metrics,
                "non_tensor_metrics": sample_non_tensor_metrics,
            }

        return reward_tensor, acc_reward_tensor, format_reward_tensor, overlong_reward_tensor, sample_metrics, invalid_uids
