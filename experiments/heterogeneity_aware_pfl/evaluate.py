"""GPT-based 7-metric evaluation and response generation."""

import os
import time
from typing import Dict, List, Optional

import openai
import torch
from datasets import Dataset
from tqdm import tqdm

from .config import EvalConfig


def _gpt_with_retry(fn, max_retries: int = 5):
    """Call fn(), retrying on transient OpenAI errors with exponential backoff."""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            return fn()
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ) as e:
            if attempt == max_retries - 1:
                raise
            print(f"\n  [GPT retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}. "
                  f"Retrying in {delay:.0f}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

# OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# MentalChat evaluation prompt: 7 metrics on 1-10 scale
_SYSTEM_PROMPT = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health counseling assistant to the user question displayed below.
\u2022 Your evaluation should be based solely on the consultation metrics defined below. Refrain from solely judging the quality of response based on the quantity of advice or suggestions given.
\u2022 Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment.
\u2022 Do not allow the length of the responses to influence your evaluation.
\u2022 Do not favor certain names of the assistants.
\u2022 Be as objective as possible.
\u2022 You must rate the response on a scale of 1 to 10 by strictly following the given format below.
\u2022 You must also rate the response in terms of EACH consultation metric defined below on a scale of 1 to 10 by strictly following the given format below.
\u2022 The ratings do not necessarily need to be the same.
\u2022 NOTE: Metrics 1\u20137 use the General Scoring Rubric below. Metric 8 (Personalization & Contextual Adaptation) uses its own distinct scoring rubric and must be scored accordingly.

IMPORTANT: Your output MUST strictly follow this format for every evaluation, so it can be parsed automatically:

Overall Rating: [number]
Active Listening: [number]
Empathy & Validation: [number]
Safety & Trustworthiness: [number]
Open-mindedness & Non-judgment: [number]
Clarity & Encouragement: [number]
Boundaries & Ethical: [number]
Holistic Approach: [number]
Personalization & Contextual Adaptation: [number]

Consultation Metrics:
Active Listening: Responses demonstrate careful consideration of user concerns, reflecting understanding and capturing the essence of the issue. Avoid assumptions or jumping to conclusions.
Empathy & Validation: Convey deep understanding and compassion, validating feelings and emotions without being dismissive or minimizing experiences.
Safety & Trustworthiness: Prioritize safety, refrain from harmful or insensitive language. Ensure the information provided is consistent and trustworthy.
Open-mindedness & Non-judgment: Approach without bias or judgment. Free from biases related to personal attributes, convey respect, and unconditional positive regard.
Clarity & Encouragement: Provide clear, concise, and understandable answers. Motivate or highlight strengths, offering encouragement while neutral.
Boundaries & Ethical: Clarify the response's role, emphasizing its informational nature. In complex scenarios, guide users to seek professional assistance.
Holistic Approach: Be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query.
Personalization & Contextual Adaptation: The response is tailored to this specific user rather than a generic mental-health audience. It (a) explicitly acknowledges details of the user's stated situation, history, stressors, or relationship to the issue; (b) adapts tone, depth, vocabulary, and therapeutic framing to fit the user's apparent emotional state and prior context; (c) avoids generic, boilerplate, or one-size-fits-all advice (e.g., bare "try mindfulness," "talk to a professional") that could be sent verbatim to any other user; and (d) integrates information from the user's question rather than ignoring it. A response that is empathic but interchangeable across users should score in the middle of the scale; a response that is uniquely useful to this user should score high.

General Scoring Rubric (Metrics 1\u20137):
1: The response completely fails to address the metric, showing a total disregard for the user's needs or concerns in this area.
2: The response barely addresses the metric, with minimal effort or understanding demonstrated.
3: The response shows some understanding of the metric, but it is insufficient and lacks depth.
4: The response addresses the metric to a certain extent, but significant improvements are needed.
5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.
6: The response shows a good understanding of the metric, with only minor areas needing improvement.
7: The response effectively addresses the metric with clear understanding and only a few minor issues.
8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.
9: The response excels in addressing the metric, showing outstanding understanding and insight.
10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness.

Personalization & Contextual Adaptation Scoring Rubric (Metric 8 only):
1-2: Response is fully generic; no reference to the user's situation; could be copy-pasted to any user with the same topic.
3-4: Single shallow acknowledgment of topic (e.g., "I hear you're feeling anxious") but content is otherwise template-like.
5-6: Multiple references to user's stated context, but tone/depth/framing is still average; some boilerplate persists.
7-8: Response clearly draws on the user's specific stressors/wording, adapts tone and depth accordingly, and avoids boilerplate; a different user with a different situation would receive a meaningfully different response.
9-10: Response is densely contextualized \u2014 names specific user-stated facts, mirrors user's vocabulary/register, and selects a therapeutic framing that fits this user's apparent emotional state. Could not plausibly be sent to a different user.

"""

# Metrics to include in composite score (excludes Overall Rating)
COMPOSITE_METRICS = [
    "Active Listening",
    "Empathy & Validation",
    "Safety & Trustworthiness",
    "Open-mindedness & Non-judgment",
    "Clarity & Encouragement",
    "Boundaries & Ethical",
    "Holistic Approach",
    "Personalization & Contextual Adaptation",
]


def _sanitize_for_api(text: str) -> str:
    """Ensure text is valid UTF-8 and free of control chars."""
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = "".join(c for c in text if c != "\x00" and (ord(c) >= 32 or c in "\n\r\t"))
    return text


def _to_ascii_safe(text: str) -> str:
    return "".join(c if ord(c) < 128 else " " for c in text)


def parse_evaluation(evaluation_text: str) -> Dict[str, int]:
    """Parse evaluation text and extract scores."""
    scores = {}
    for line in evaluation_text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            metric = parts[0].strip()
            try:
                score = int(parts[1].strip())
                scores[metric] = score
            except (ValueError, IndexError):
                continue
    return scores


def gpt_evaluate_single(
    user_question: str, model_response: str, eval_config: EvalConfig
) -> Dict[str, int]:
    """Call GPT to evaluate a single Q/A pair. Returns metric scores dict."""
    user_question = _sanitize_for_api(user_question)
    model_response = _sanitize_for_api(model_response)
    content = (
        f"[User Question]\n{user_question}\n\n"
        f"[The Start of AI assistant's Answer]\n{model_response}\n"
        f"[The End of AI assistant's Answer]"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    client = openai.OpenAI()
    try:
        response = _gpt_with_retry(
            lambda: client.chat.completions.create(
                model=eval_config.model,
                messages=messages,
                temperature=0,
                seed=42,
            )
        )
        return parse_evaluation(response.choices[0].message.content)
    except openai.BadRequestError as e:
        if "parse the json" in str(e).lower():
            messages_ascii = [
                {"role": "system", "content": _to_ascii_safe(_SYSTEM_PROMPT)},
                {"role": "user", "content": _to_ascii_safe(content)},
            ]
            response = _gpt_with_retry(
                lambda: client.chat.completions.create(
                    model=eval_config.model,
                    messages=messages_ascii,
                    temperature=0,
                    seed=42,
                )
            )
            return parse_evaluation(response.choices[0].message.content)
        raise


def generate_response(prompt: str, model, tokenizer, device: str, max_new_tokens: int = 512) -> str:
    """Generate a single response using greedy decoding."""
    if hasattr(tokenizer, "apply_chat_template") and isinstance(prompt, list):
        prompt_str = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)

    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    use_autocast = device == "cuda" and torch.cuda.is_available()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                do_sample=False,
            )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_responses(
    model,
    tokenizer,
    dataset: Dataset,
    device: str,
    max_new_tokens: int = 512,
    max_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Generate responses for a dataset. Returns list of {question, response}."""
    results = []
    n = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i in tqdm(range(n), desc="Generating responses", leave=False):
        example = dataset[i]
        question = str(example.get("input", ""))
        prompt = [{"role": "user", "content": question}]
        response = generate_response(prompt, model, tokenizer, device, max_new_tokens)
        results.append({"question": question, "response": response})

    return results


def evaluate_client(
    model,
    tokenizer,
    dataset: Dataset,
    device: str,
    eval_config: EvalConfig,
) -> Dict[str, float]:
    """Generate responses on dataset, evaluate each with GPT, return mean scores."""
    responses = generate_responses(
        model, tokenizer, dataset, device,
        max_new_tokens=eval_config.max_new_tokens,
        max_samples=eval_config.max_eval_samples,
    )

    all_scores: Dict[str, List[float]] = {}
    for item in tqdm(responses, desc="GPT evaluating", leave=False):
        scores = gpt_evaluate_single(item["question"], item["response"], eval_config)
        for metric, score in scores.items():
            all_scores.setdefault(metric, []).append(score)

    # Compute means
    mean_scores = {metric: sum(vals) / len(vals) for metric, vals in all_scores.items()}

    # Composite score: average of the 8 consultation metrics (excludes Overall Rating)
    composite_vals = [mean_scores[m] for m in COMPOSITE_METRICS if m in mean_scores]
    if composite_vals:
        mean_scores["composite"] = sum(composite_vals) / len(composite_vals)

    return mean_scores


def compute_group_averages(
    client_scores: Dict[int, Dict[str, float]],
    client_types: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    """Compute all-client avg, IID-only avg, non-IID-only avg."""
    groups = {"all": list(client_scores.keys())}
    groups["iid"] = [cid for cid, t in client_types.items() if t == "iid"]
    groups["noniid"] = [cid for cid, t in client_types.items() if t == "noniid"]

    result = {}
    for group_name, cids in groups.items():
        if not cids:
            continue
        group_scores: Dict[str, List[float]] = {}
        for cid in cids:
            if cid not in client_scores:
                continue
            for metric, val in client_scores[cid].items():
                group_scores.setdefault(metric, []).append(val)
        result[group_name] = {
            metric: sum(vals) / len(vals) for metric, vals in group_scores.items()
        }

    return result
