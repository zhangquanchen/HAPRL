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
"""Utils for tokenization."""
import json
import os
import warnings

__all__ = ['hf_tokenizer', 'hf_processor']


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def _normalize_extra_special_tokens_for_transformers(name_or_path, kwargs):
    """Compat for tokenizer_config files whose extra_special_tokens is a list."""
    if "extra_special_tokens" in kwargs or not isinstance(name_or_path, str):
        return
    config_path = os.path.join(name_or_path, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        kwargs["extra_special_tokens"] = {
            f"extra_special_token_{idx}": token for idx, token in enumerate(extra)
        }


class _SimpleQwen3VLProcessor:
    """Minimal Qwen3-VL processor for rollout code paths that need image preprocessing."""

    def __init__(self, image_processor, tokenizer, video_processor=None, chat_template=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.chat_template = chat_template

        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")

        self.image_token_id = getattr(tokenizer, "image_token_id", None) or tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = getattr(tokenizer, "video_token_id", None) or tokenizer.convert_tokens_to_ids(self.video_token)
        self.vision_start_token_id = getattr(tokenizer, "vision_start_token_id", None) or tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = getattr(tokenizer, "vision_end_token_id", None) or tokenizer.convert_tokens_to_ids(self.vision_end_token)


def _read_chat_template(name_or_path):
    if not isinstance(name_or_path, str):
        return None
    chat_template_path = os.path.join(name_or_path, "chat_template.jinja")
    if not os.path.isfile(chat_template_path):
        return None
    try:
        with open(chat_template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _load_qwen3_vl_processor_fallback(name_or_path, kwargs):
    if not isinstance(name_or_path, str):
        return None
    processor_config_path = os.path.join(name_or_path, "processor_config.json")
    if not os.path.isfile(processor_config_path):
        return None
    try:
        with open(processor_config_path, "r", encoding="utf-8") as f:
            processor_config = json.load(f)
    except Exception:
        return None
    if processor_config.get("processor_class") != "Qwen3VLProcessor":
        return None

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    image_config = dict(processor_config.get("image_processor") or {})
    image_config.pop("image_processor_type", None)
    video_config = dict(processor_config.get("video_processor") or {})
    video_config.pop("video_processor_type", None)

    tokenizer = hf_tokenizer(name_or_path, **kwargs)
    image_processor = Qwen2VLImageProcessor(**image_config)
    video_processor = Qwen3VLVideoProcessor(**video_config) if video_config else None
    return _SimpleQwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=_read_chat_template(name_or_path),
    )


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    _normalize_extra_special_tokens_for_transformers(name_or_path, kwargs)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor
    _normalize_extra_special_tokens_for_transformers(name_or_path, kwargs)
    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    if processor is None:
        processor = _load_qwen3_vl_processor_fallback(name_or_path, kwargs)
    return processor
