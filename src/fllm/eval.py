checkpoint_root = "./FL_iid_qwen0.5b_newcluster_seed75"
SEED = 75
EVAL_DATASET = "mentalchat"  # mentalchat | dolly | medmcqa
# export API KEY!!!!

import openai
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from collections import defaultdict
from tqdm import tqdm

# Import shared split logic (same as fed.py and generate.py)
try:
    from fllm.fed import get_train_test_indices, CLUSTERED_CSV_PATH
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fllm.fed import get_train_test_indices, CLUSTERED_CSV_PATH

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

_SYSTEM_PROMPT_MENTALCHAT = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health counseling assistant to the user question displayed below.
• Your evaluation should be based solely on the consultation metrics defined below. Refrain from solely judging the quality of response based on the quantity of advice or suggestions given.
• Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment.
• Do not allow the length of the responses to influence your evaluation.
• Do not favor certain names of the assistants.
• Be as objective as possible.
• You must rate the response on a scale of 1 to 10 by strictly following the given format below.
• You must also rate the response in terms of EACH consultation metric defined below on a scale of 1 to 10 by strictly following the given format below.
• The ratings do not necessarily need to be the same.
• NOTE: Metrics 1–7 use the General Scoring Rubric below. Metric 8 (Personalization & Contextual Adaptation) uses its own distinct scoring rubric and must be scored accordingly.

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
Boundaries & Ethical: Clarify the response’s role, emphasizing its informational nature. In complex scenarios, guide users to seek professional assistance.
Holistic Approach: Be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query.
Personalization & Contextual Adaptation: The response is tailored to this specific user rather than a generic mental-health audience. It (a) explicitly acknowledges details of the user’s stated situation, history, stressors, or relationship to the issue; (b) adapts tone, depth, vocabulary, and therapeutic framing to fit the user’s apparent emotional state and prior context; (c) avoids generic, boilerplate, or one-size-fits-all advice (e.g., bare "try mindfulness," "talk to a professional") that could be sent verbatim to any other user; and (d) integrates information from the user’s question rather than ignoring it. A response that is empathic but interchangeable across users should score in the middle of the scale; a response that is uniquely useful to this user should score high.

General Scoring Rubric (Metrics 1–7):
1: The response completely fails to address the metric, showing a total disregard for the user’s needs or concerns in this area.
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
1-2: Response is fully generic; no reference to the user’s situation; could be copy-pasted to any user with the same topic.
3-4: Single shallow acknowledgment of topic (e.g., "I hear you’re feeling anxious") but content is otherwise template-like.
5-6: Multiple references to user’s stated context, but tone/depth/framing is still average; some boilerplate persists.
7-8: Response clearly draws on the user’s specific stressors/wording, adapts tone and depth accordingly, and avoids boilerplate; a different user with a different situation would receive a meaningfully different response.
9-10: Response is densely contextualized — names specific user-stated facts, mirrors user’s vocabulary/register, and selects a therapeutic framing that fits this user’s apparent emotional state. Could not plausibly be sent to a different user.

"""

_SYSTEM_PROMPT_DOLLY = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below.
• Your evaluation should be based solely on the instruction-following metrics defined below.
• Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment.
• Do not allow the length of the responses to influence your evaluation.
• Be as objective as possible.
• You must rate the response on a scale of 1 to 10 by strictly following the given format below.
• You must also rate the response in terms of EACH metric defined below on a scale of 1 to 10 by strictly following the given format below.
• The ratings do not necessarily need to be the same.

IMPORTANT: Your output MUST strictly follow this format for every evaluation, so it can be parsed automatically:

Overall Rating: [number]
Task Completion: [number]
Accuracy & Correctness: [number]
Clarity & Coherence: [number]
Relevance: [number]
Helpfulness: [number]
Depth & Detail: [number]

Instruction-Following Metrics:
Task Completion: The response fully addresses what the instruction asks for, without omitting key requirements or adding irrelevant content.
Accuracy & Correctness: The information provided is factually correct and free from significant errors or misleading statements.
Clarity & Coherence: The response is well-structured, easy to read, and logically organized. Ideas flow naturally and the language is clear.
Relevance: The response stays on-topic and directly addresses the instruction without unnecessary tangents.
Helpfulness: The response is practically useful to the user. It provides actionable, substantive value rather than vague or generic answers.
Depth & Detail: The level of detail is appropriate for the task. Complex instructions receive thorough treatment; simple ones are answered concisely without padding.

Scoring rubric:
1: The response completely fails to address the metric, showing a total disregard for the user’s needs or concerns in this area.
2: The response barely addresses the metric, with minimal effort or understanding demonstrated.
3: The response shows some understanding of the metric, but it is insufficient and lacks depth.
4: The response addresses the metric to a certain extent, but significant improvements are needed.
5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.
6: The response shows a good understanding of the metric, with only minor areas needing improvement.
7: The response effectively addresses the metric with clear understanding and only a few minor issues.
8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.
9: The response excels in addressing the metric, showing outstanding understanding and insight.
10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness.

"""

_SYSTEM_PROMPT_MEDMCQA = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the medical multiple-choice question displayed below.
• Your evaluation should be based solely on the medical QA metrics defined below.
• Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment.
• Do not allow the length of the responses to influence your evaluation.
• Be as objective as possible.
• You must rate the response on a scale of 1 to 10 by strictly following the given format below.
• You must also rate the response in terms of EACH metric defined below on a scale of 1 to 10 by strictly following the given format below.
• The ratings do not necessarily need to be the same.

IMPORTANT: Your output MUST strictly follow this format for every evaluation, so it can be parsed automatically:

Overall Rating: [number]
Answer Correctness: [number]
Explanation Quality: [number]
Medical Accuracy: [number]
Clarity: [number]
Completeness: [number]
Educational Value: [number]

Medical QA Metrics:
Answer Correctness: The response selects the correct answer option among the given choices. Partial credit if the reasoning is sound but the final selection is wrong.
Explanation Quality: The explanation supporting the answer is logical, well-reasoned, and follows sound clinical or scientific reasoning.
Medical Accuracy: All medical facts, terminology, and clinical information stated in the response are accurate and consistent with established medical knowledge.
Clarity: The response is clearly written, well-organized, and easy to follow. Technical terms are used appropriately without being unnecessarily obscure.
Completeness: The response fully addresses the question, including both the answer selection and a sufficient explanation. It does not leave key aspects unanswered.
Educational Value: The response helps the reader understand the underlying medical concept, not just the answer. It provides context or insight that aids learning.

Scoring rubric:
1: The response completely fails to address the metric, showing a total disregard for the user’s needs or concerns in this area.
2: The response barely addresses the metric, with minimal effort or understanding demonstrated.
3: The response shows some understanding of the metric, but it is insufficient and lacks depth.
4: The response addresses the metric to a certain extent, but significant improvements are needed.
5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.
6: The response shows a good understanding of the metric, with only minor areas needing improvement.
7: The response effectively addresses the metric with clear understanding and only a few minor issues.
8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.
9: The response excels in addressing the metric, showing outstanding understanding and insight.
10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness.

"""

_PROMPTS = {
    "mentalchat": _SYSTEM_PROMPT_MENTALCHAT,
    "dolly":      _SYSTEM_PROMPT_DOLLY,
    "medmcqa":    _SYSTEM_PROMPT_MEDMCQA,
}
system_prompt = _PROMPTS.get(EVAL_DATASET, _SYSTEM_PROMPT_MENTALCHAT)


def _sanitize_for_api(text: str) -> str:
    """Ensure text is valid UTF-8 and free of control chars that can break JSON/API requests."""
    if not isinstance(text, str):
        text = str(text)
    # Encode to UTF-8, replacing invalid code points, then decode
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # Remove null bytes and other control characters that can break JSON
    text = "".join(c for c in text if c != "\x00" and (ord(c) >= 32 or c in "\n\r\t"))
    return text


def _to_ascii_safe(text: str) -> str:
    """Replace non-ASCII with space so JSON body is ASCII-only (fallback for API parse errors)."""
    return "".join(c if ord(c) < 128 else " " for c in text)


def gpt_evaluate(user_question, model_response):
    user_question = _sanitize_for_api(user_question)
    model_response = _sanitize_for_api(model_response)
    content = f"[User Question]\n{user_question}\n\n[The Start of AI assistant's Answer]\n{model_response}\n[The End of AI assistant's Answer]"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
        )
        return response.choices[0].message.content
    except openai.BadRequestError as e:
        if "parse the json" in str(e).lower():
            messages_ascii = [
                {"role": "system", "content": _to_ascii_safe(system_prompt)},
                {"role": "user", "content": _to_ascii_safe(content)},
            ]
            print(f"BadRequestError: {e}")
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages_ascii,
            )
            return response.choices[0].message.content
        raise


def parse_evaluation(evaluation_text):
    """Parse evaluation text and extract scores."""
    scores = {}
    lines = evaluation_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            metric = parts[0].strip()
            try:
                score = int(parts[1].strip())
                scores[metric] = score
            except (ValueError, IndexError):
                continue
    
    return scores


def calculate_mean_scores(results):
    """
    Calculate mean scores for each metric across all results.
    
    Args:
        results: List of dictionaries with scores
    
    Returns:
        Dictionary with mean scores for each metric
    """
    metric_sums = defaultdict(list)
    
    # Collect all scores for each metric
    for result in results:
        for key, value in result.items():
            if key not in ["question", "response"] and isinstance(value, (int, float)):
                metric_sums[key].append(value)
    
    # Calculate means
    mean_scores = {}
    for metric, scores in metric_sums.items():
        if scores:
            mean_scores[f"mean_{metric}"] = sum(scores) / len(scores)
    
    return mean_scores


def evaluate_dataset(test_data):
    """
    Evaluate a dataset and return a list of dictionaries.
    Each dictionary contains question, response, and each score.
    
    Args:
        test_data: List of dictionaries with 'question' and 'response' keys
    
    Returns:
        List of dictionaries with question, response, and all scores
    """
    results = []
    
    print(f"\nEvaluating {len(test_data)} examples...")
    for idx, example in enumerate(tqdm(test_data, desc="Evaluating", unit="example")):
        user_question = example["question"]
        model_response = example["response"]
        evaluation_text = gpt_evaluate(user_question, model_response)
        scores = parse_evaluation(evaluation_text)
        
        # Create result dictionary
        result = {
            "question": user_question,
            "response": model_response,
            **scores  # Unpack all scores into the dictionary
        }
        results.append(result)
    
    return results


def load_existing_results(path: Path):
    """Load existing eval results; expect a JSON array of result dicts (no mean_scores)."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return []


def load_human_responses_from_dataset(
    dataset_name: str = "ShenLab/MentalChat16K",
    split: str = "train",
    max_examples: Optional[int] = None,
    test_size: float = 0.02,
    seed: int = SEED,
):
    """
    Load question and human response pairs from clustered_dataset.csv (same data source and
    split method as fed.py and generate.py for identical train/test split).
    Returns a list of dicts with 'question' and 'response' keys, matching the format
    expected by evaluate_dataset().
    """
    if not os.path.isfile(CLUSTERED_CSV_PATH):
        raise FileNotFoundError(
            f"Clustered dataset CSV not found at {CLUSTERED_CSV_PATH}. "
            "Run clustering.py to create clustered_dataset.csv first."
        )
    df = pd.read_csv(CLUSTERED_CSV_PATH)
    if test_size is None or not (0 < test_size < 1):
        test_df = df
    else:
        n_total = len(df)
        _, test_indices = get_train_test_indices(n_total, test_size, seed)
        test_df = df.iloc[test_indices]

    # CSV has input (question) and output (response)
    question_col = "input" if "input" in test_df.columns else "instruction"
    response_col = "output" if "output" in test_df.columns else "response"
    if question_col not in test_df.columns or response_col not in test_df.columns:
        raise ValueError(
            f"CSV must have 'input' and 'output' columns (found: {list(test_df.columns)})"
        )

    data = []
    for _, row in test_df.iterrows():
        q = row.get(question_col, "")
        r = row.get(response_col, "")
        if q:
            data.append({"question": str(q).strip(), "response": str(r).strip()})
        if max_examples is not None and len(data) >= max_examples:
            break

    return data


def load_generated_file(path: Path):
    """
    Load generated file (array of {question, response}).
    Generate writes the array incrementally and only adds the closing ']' at the end,
    so we read as text and close the array if needed to parse mid-generation.
    """
    with open(path, "r") as f:
        raw = f.read()
    raw = raw.rstrip()
    if raw and not raw.endswith("]"):
        if raw.endswith(","):
            raw = raw[:-1].rstrip()
        raw += "\n]"
    if not raw or raw.strip() == "]":
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def main():
    parser = argparse.ArgumentParser(description="Evaluate new examples from generated output or original dataset human responses.")
    parser.add_argument("--generated", default=f"generated_output6/{checkpoint_root}.json", help="Path to generated JSON (array of {question, response})")
    parser.add_argument("--output", default=f"eval_results7/{checkpoint_root}.json", help="Path to eval output (JSON array of results, no average)")
    parser.add_argument("--last", type=int, default=None, help="Evaluate only the last N examples (default: all new since last run)")
    parser.add_argument(
        "--eval_original_dataset",
        action="store_true",
        help="Evaluate human responses from the original dataset (MentalChat16K) instead of model-generated output",
    )
    parser.add_argument(
        "--dataset_name",
        default="ShenLab/MentalChat16K",
        help="Dataset name when using --eval_original_dataset (default: ShenLab/MentalChat16K)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split when using --eval_original_dataset (default: train)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.02,
        help="Test set fraction when using --eval_original_dataset (same as generate.py, default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for train/test split when using --eval_original_dataset (default: 42)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples to evaluate when using --eval_original_dataset (default: all)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_original_dataset:
        print(f"Loading human responses from dataset: {args.dataset_name} (split={args.split})")
        to_eval = load_human_responses_from_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            max_examples=args.max_examples,
            test_size=args.test_size,
            seed=args.seed,
        )
        existing_results = []
        n_existing = 0
        if not to_eval:
            print("No examples with valid question/response found in dataset.")
            return
        print(f"Loaded {len(to_eval)} human response examples for evaluation.")
    else:
        generated_path = Path(args.generated)
        if not generated_path.exists():
            print(f"Generated file not found: {generated_path}", file=sys.stderr)
            sys.exit(1)
        generated_data = load_generated_file(generated_path)
        existing_results = load_existing_results(output_path)
        n_existing = len(existing_results)

        if args.last is not None:
            to_eval = generated_data[-args.last:]
        else:
            to_eval = generated_data[n_existing:]
        if not to_eval:
            print("No new examples to evaluate.")
            return
        print(f"Evaluating {len(to_eval)} new examples (existing: {n_existing}, total in generated: {len(generated_data)}).")

    new_results = evaluate_dataset(to_eval)
    all_results = existing_results + new_results

    # Save as JSON array only (no mean_scores / no average at the end)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Evaluation results saved to {output_path} ({len(all_results)} total, no average in file).")
    print(f"Evaluated {len(new_results)} new examples.")
    mean_scores = calculate_mean_scores(new_results)
    print("Mean scores for this batch:")
    for metric, score in sorted(mean_scores.items()):
        print(f"  {metric}: {score:.2f}")


if __name__ == "__main__":
    main()