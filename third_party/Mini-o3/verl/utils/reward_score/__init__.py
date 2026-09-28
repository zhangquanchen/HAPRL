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
# from . import gsm8k, math, prime_math, prime_code

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
        'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12', 'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ["vstar_bench", "visual_probe_easy", "visual_probe_medium", "visual_probe_hard", "visual_probe_train", "deepeyes_train"]:
        reward_fn_name = extra_info['general_qa_reward_fn']
        if reward_fn_name == 'general_qa_tool':
            from . import general_qa_tool
            res = general_qa_tool.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'general_qa_tool_qwen':
            from . import general_qa_tool_qwen
            res = general_qa_tool_qwen.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'general_qa_tool_mc':
            from . import general_qa_tool_mc
            res = general_qa_tool_mc.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_data':
            from . import qa_data
            res = qa_data.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_process':
            from . import qa_process
            res = qa_process.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_process_v1':
            from . import qa_process_v1
            res = qa_process_v1.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'rollout_noop':
            from . import rollout_noop
            res = rollout_noop.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_process_v2':
            from . import qa_process_v2
            res = qa_process_v2.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_process_v3':
            from . import qa_process_v3
            res = qa_process_v3.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        else:
            raise NotImplementedError(f"Unknown reward_fn '{reward_fn_name}' for data_source '{data_source}'")
    elif data_source in ["iou_data"]:
        from . import iou_bbox
        res = iou_bbox.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
    elif data_source in ["iou_process"]:
        from . import iou_process
        res = iou_process.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
    elif data_source in ["qa_data"]:
        from . import qa_data
        res = qa_data.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
    elif data_source in ["qa_process"]:
        from . import qa_process
        res = qa_process.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
    elif data_source in ["qa_process_v1"]:
        reward_fn_name = (extra_info or {}).get('general_qa_reward_fn')
        if reward_fn_name == 'rollout_noop':
            from . import rollout_noop
            res = rollout_noop.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_process_v3':
            from . import qa_process_v3
            res = qa_process_v3.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
        elif reward_fn_name == 'qa_process_v2':
            from . import qa_process_v2
            res = qa_process_v2.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
        else:
            from . import qa_process_v1
            res = qa_process_v1.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
    elif data_source in ["qa_process_v2"]:
        from . import qa_process_v2
        res = qa_process_v2.compute_score(kwargs["prompt"], solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError
    return res
