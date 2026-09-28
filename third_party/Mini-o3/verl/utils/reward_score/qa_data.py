# verl/utils/reward_score/qa_data.py
import os
import re
import time
import openai

SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\n"
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully.\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:

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

def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default

class InternalOpenAIClient:
    """
    内网 GPT 网关 client（sankuai AIGC）
    依赖环境变量：
      - INTERNAL_OPENAI_API_KEY (必填)
      - INTERNAL_OPENAI_BASE_URL (默认 https://aigc.sankuai.com/v1/openai/native)
      - INTERNAL_OPENAI_MODEL (默认 gpt-4.1)
      - INTERNAL_OPENAI_TRACE_ID (可选，用于 header)
      - QA_GPT_TIMEOUT_S (可选，默认 60)
      - QA_GPT_MAX_RETRIES (可选，默认 3)
      - QA_GPT_BACKOFF_S (可选，默认 2)
    """

    def __init__(self):
        api_key = _env("INTERNAL_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Need env INTERNAL_OPENAI_API_KEY for qa_baseline reward.")
        base_url = _env("INTERNAL_OPENAI_BASE_URL", "https://aigc.sankuai.com/v1/openai/native")
        self.model = _env("INTERNAL_OPENAI_MODEL", "gpt-4.1")
        self.trace_id = _env("INTERNAL_OPENAI_TRACE_ID", "")
        self.timeout_s = float(_env("QA_GPT_TIMEOUT_S", "60"))
        self.max_retries = int(_env("QA_GPT_MAX_RETRIES", "3"))
        self.backoff_s = float(_env("QA_GPT_BACKOFF_S", "2"))

        # OpenAI SDK (non-Azure) client
        # 注：header 是否生效取决于你们网关实现；不支持也不会影响功能
        extra_headers = {}
        if self.trace_id:
            extra_headers["x-trace-id"] = self.trace_id

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers if extra_headers else None,
        )

    def query(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=min(0.2 * attempt, 1.0),
                    max_tokens=256,   # judge 只要输出 "Score: 0/1"，不用很大
                    timeout=self.timeout_s,
                )
                text = (resp.choices[0].message.content or "").strip()
                if "score:" not in text.lower():
                    raise ValueError(f"No 'Score:' in response: {text[:200]}")
                # 提取 score token
                extract = text.lower().split("score:")[-1].strip().splitlines()[0].strip().split(" ")[0]
                if ("1" not in extract) and ("0" not in extract):
                    raise ValueError(f"Bad score token: {extract}")
                return extract
            except Exception as e:
                # 最后一次失败才打印详细日志，避免刷屏
                if attempt == self.max_retries - 1:
                    print("[qa_baseline][judge_error]", repr(e))
                    print("[qa_baseline][judge_prompt]\n", prompt[:1500])
                time.sleep(self.backoff_s)
        return ""

client = None

# -------------------------
# format checks（保持与 general_qa_tool.py 一致）
# -------------------------
def grounding_format_reward(predict_str_list, extra_info=None):
    """
    返回 (format_score, tool_call_count)
    - 多轮：前N-1轮必须 <think>... + <grounding>...；最后一轮 <think>... + <answer>...
    - 单轮：必须 <think>... + <answer>... 且不含 <grounding>
    """
    step_grounding_format = r"^<think>.*?</think>\s*<grounding>.*?</grounding>\s*$"
    direct_answer_format = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"

    conv_rounds = len(predict_str_list)
    tool_call_count = 0

    if conv_rounds == 1:
        resp = predict_str_list[0]
        ok = re.match(direct_answer_format, resp, re.DOTALL) is not None
        # direct answer 不允许 grounding
        if "<grounding>" in resp or "</grounding>" in resp:
            ok = False
        return (1.0 if ok else 0.0), 0

    # multi-turn
    ok = True
    for i in range(conv_rounds - 1):
        r = predict_str_list[i]
        if re.match(step_grounding_format, r, re.DOTALL) is None:
            ok = False
            break
        if r.count("<grounding>") >= 1:
            tool_call_count += 1
        # 中间轮不允许 answer
        if "<answer>" in r or "</answer>" in r:
            ok = False
            break

    last = predict_str_list[-1]
    if re.match(direct_answer_format, last, re.DOTALL) is None:
        ok = False
    # 最后一轮不允许 grounding
    if "<grounding>" in last or "</grounding>" in last:
        ok = False

    return (1.0 if ok else 0.0), tool_call_count

# -------------------------
# accuracy reward（GPT judge）
# -------------------------
def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str, gpt_extract_answer=False, extra_info=None):
    global client
    if client is None:
        client = InternalOpenAIClient()

    original_predict_str = " ".join(predict_str_list)

    # 只评价最终答案：抽 <answer>...</answer>
    if gpt_extract_answer:
        mode = extra_info.get("extract_answer_tags", "strict") if extra_info else "strict"
        if mode == "split":
            original_predict_str = original_predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
        elif mode == "strict":
            m = re.search(r"<answer>(.*?)</answer>", original_predict_str, re.DOTALL)
            if not m:
                return 0.0
            original_predict_str = m.group(1).strip()
        elif mode == "strict_v2":
            m = re.search(r"<answer>(.*?)</answer>\s*$", original_predict_str, re.DOTALL)
            if not m:
                return 0.0
            original_predict_str = m.group(1).strip()
        else:
            raise ValueError(f"Unknown extract_answer_tags={mode}")

    judge_prompt = QUERY_PROMPT.format(question=prompt, ground_truth=original_answer, prediction=original_predict_str)
    resp = client.query(prompt=judge_prompt, system_prompt=SYSTEM_PROMPT)

    if not resp:
        return {"is_filter": True, "info": "error with internal gpt judge"}
    return 1.0 if "1" in resp else 0.0

def acc_reward(prompt: str, predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False) if extra_info else False
    return inner_acc_reward(prompt, predict_str_list, solution, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)

def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    acc_reward_weight = extra_info.get("acc_reward_weight", 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get("format_reward_weight", 1.0) if extra_info else 1.0

    tool_call_penalty = 0.1
    if extra_info is not None and "tool_call_penalty" in extra_info:
        tool_call_penalty = extra_info.get("tool_call_penalty", 0.1)

    acc = acc_reward(prompt, predict_str_list, ground_truth, extra_info)
    if isinstance(acc, dict):
        return acc

    format_score_raw, tool_call_count = grounding_format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score_raw

    tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
    tool_reward = extra_info.get("use_tool_reward_weight", 0.0) if (extra_info and tool_call_count > 0) else 0.0

    score = tool_penalty_factor * acc_score + format_score + tool_reward
    return score, acc_score, format_score
