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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.constants import SYSTEM_PROMPT_MAP

import ray
import hydra
import os


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)

#----------
def run_ppo(config, compute_score=None):
    import os
    import ray

    if not ray.is_initialized():
        # 在共享账号场景，必须显式连接脚本启动的 head，禁止 auto
        addr = os.environ.get("RAY_ADDRESS", "").strip()
        assert addr, "RAY_ADDRESS is required (do not use auto in a shared account)."

        runtime_env_vars = {
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_DEBUG": "WARN",
            "RAY_USAGE_STATS_ENABLED": "0",
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "0"),
        }
        for key in (
            "RAY_ADDRESS",
            "RAY_NAMESPACE",
            "RAY_TMPDIR",
            "VLLM_WORKER_MULTIPROC_METHOD",
            "VLLM_DISABLE_USAGE_STATS",
            "VLLM_USAGE_STATS_ENABLED",
            "VLLM_NO_USAGE_STATS",
            "VLLM_USE_CUMEM",
            "VLLM_DISABLE_CUMEM",
            "VLLM_CUDA_MEM_POOL_SIZE",
            "INTERNAL_OPENAI_API_KEY",
            "INTERNAL_OPENAI_BASE_URL",
            "INTERNAL_OPENAI_MODEL",
            "INTERNAL_OPENAI_VERIFY_SSL",
            "INTERNAL_OPENAI_TRANSPORT",
            "INTERNAL_OPENAI_EXTRA_BODY_JSON",
            "QA_PROCESS_TOTAL_MODE",
            "VERL_GLOBAL_POOL_ID",
        ):
            value = os.environ.get(key)
            if value is not None:
                runtime_env_vars[key] = value

        ray.init(
            address=addr,
            namespace=os.environ.get("RAY_NAMESPACE", "minio3"),
            ignore_reinit_error=True,
            include_dashboard=False,
            _temp_dir=os.environ.get("RAY_TMPDIR", "/tmp/ray_minio3"),
            runtime_env={"env_vars": runtime_env_vars},
        )

    ray.get(main_task.remote(config, compute_score))

#----------
"""
def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
       
        #ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        ray.init(
            include_dashboard=False,
            _temp_dir="/tmp/ray",
            num_cpus=4,
            object_store_memory=512 * 1024 * 1024,
            ignore_reinit_error=True,
            log_to_driver=True,
            #={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN',
            #                       'RAY_USAGE_STATS_ENABLED': '0'}}
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true',
                          'NCCL_DEBUG': 'WARN',
                          'RAY_USAGE_STATS_ENABLED': '0',
                          'VLLM_USE_V1': os.environ.get("VLLM_USE_V1", "0")}}
        )
        
        #----------------
        # 只连接由 sh 启动的 head（避免 Python 自己再起一套 Ray）
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            namespace=os.environ.get("RAY_NAMESPACE", "minio3"),
            ignore_reinit_error=True,
        )
        #---------------------
    ray.get(main_task.remote(config, compute_score))
"""
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    # from omegaconf import OmegaConf

    from omegaconf import OmegaConf,open_dict
    with open_dict(config["actor_rollout_ref"]):
        OmegaConf.update(config["actor_rollout_ref"], "data", config["data"], merge=False)
        OmegaConf.update(config["actor_rollout_ref"], "reward_model", config["reward_model"], merge=False)

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
        #from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker)
    }

    global_pool_id = os.environ.get('VERL_GLOBAL_POOL_ID', 'global_pool')
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id
    }

    if config.actor_rollout_ref.actor.use_kl_loss and config.actor_rollout_ref.actor.kl_loss_coef > 0:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'naive_multithreads_tool':
        from verl.workers.reward_manager import NaiveMultiThreadsToolRewardManager
        reward_manager_cls = NaiveMultiThreadsToolRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError
    
    config.actor_rollout_ref.rollout.max_pixels = config.data.max_pixels
    config.actor_rollout_ref.rollout.min_pixels = config.data.min_pixels

    system_prompt_type = config.data.get('system_prompt', False)
    system_prompt = SYSTEM_PROMPT_MAP[system_prompt_type] if system_prompt_type else None
    extra_info = {
        "acc_reward_weight": config.data.get("acc_reward_weight", 1.0),
        "format_reward_weight": config.data.get("format_reward_weight", 1.0),
        "use_tool_reward_weight": config.data.get("use_tool_reward_weight", 0.0),
        "tool_call_penalty": config.data.get("tool_call_penalty", 0.1),
        "extract_answer_tags": config.data.get("extract_answer_tags", "split"),
        "general_qa_reward_fn": config.data.get("general_qa_reward_fn", "v1"),
        "gpt_extract_answer": config.data.get("gpt_extract_answer", False),
        "model_system_prompt": system_prompt,
        "max_total_response_length": config.actor_rollout_ref.rollout.max_total_response_length,
        "overlong_buffer_len": config.reward_model.get("overlong_buffer_len", 0),
        #------
        "process_reward_weight": config.data.get("process_reward_weight", 0.0),
        # qa_process_v1: if process_score < threshold, total reward -> 0; null disables
        "process_reward_gate_threshold": config.data.get("process_reward_gate_threshold", 0.3),
        #------
    }
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, mode="train", extra_info=extra_info, gpt_threads=config.data.get("gpt_threads", 100))

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=extra_info, gpt_threads=config.data.get("gpt_threads", 100))

    use_hybrid_reward_manager = config.reward_model.get("use_hybrid_reward_manager", False)
    if use_hybrid_reward_manager:
        from verl.workers.reward_manager import NaiveMultiThreadsToolRewardManager
        import copy
        gpt_extra_info = copy.deepcopy(extra_info)
        gpt_extra_info['general_qa_reward_fn'] = config.data.get(
            "gpt_general_qa_reward_fn",
            config.data.get("general_qa_reward_fn", "general_qa_tool"),
        )
        gpt_reward_fn = NaiveMultiThreadsToolRewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, mode="train", extra_info=gpt_extra_info, gpt_threads=config.data.get("gpt_threads", 100))
        val_gpt_reward_fn = NaiveMultiThreadsToolRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=gpt_extra_info, gpt_threads=config.data.get("gpt_threads", 100))

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            gpt_reward_fn=gpt_reward_fn if use_hybrid_reward_manager else None,
                            val_gpt_reward_fn=val_gpt_reward_fn if use_hybrid_reward_manager else None)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
