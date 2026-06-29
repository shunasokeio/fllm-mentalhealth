import json
import csv
from pathlib import Path

from datasets import load_dataset


def load_eval_results(file_path):
    """Load evaluation results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_human_responses_from_hf(
    dataset_name: str = "ShenLab/MentalChat16K",
    split: str = "train",
) -> dict:
    """
    Load human responses from a Hugging Face dataset.

    Assumes the dataset has:
      - either an 'input' or 'question' column (used as the question text)
      - either a 'human_response' column (preferred) or an 'output' column

    Returns:
        Dict mapping question -> human_response text
    """
    ds = load_dataset(dataset_name, split=split)

    # Determine which column holds the question text
    if "question" in ds.column_names:
        question_col = "question"
    elif "input" in ds.column_names:
        question_col = "input"
    else:
        raise ValueError(
            f"Neither 'question' nor 'input' column found in dataset {dataset_name}"
        )

    # Prefer 'human_response', fall back to 'output' if needed (MentalChat16K uses 'output')
    if "human_response" in ds.column_names:
        response_col = "human_response"
    elif "output" in ds.column_names:
        response_col = "output"
    else:
        raise ValueError(
            f"Neither 'human_response' nor 'output' column found in dataset {dataset_name}"
        )

    mapping: dict[str, str] = {}
    for row in ds:
        q = row[question_col]
        r = row[response_col]
        # If there are duplicate questions, last one wins; this is fine for our use
        mapping[q] = r

    return mapping


def create_comparison_csv(base_file, fine_tuned_file, output_csv_path):
    """
    Create a CSV file comparing base, fine-tuned, and human responses.

    Args:
        base_file: Path to base model evaluation results JSON file
        fine_tuned_file: Path to fine-tuned model evaluation results JSON file
        output_csv_path: Path to output CSV file
    """
    # Load both JSON files
    base_data = load_eval_results(base_file)
    fine_tuned_data = load_eval_results(fine_tuned_file)

    base_results = base_data["results"]
    fine_tuned_results = fine_tuned_data["results"]

    # Load human responses from HF dataset
    human_responses = load_human_responses_from_hf()

    # Match results by index (assuming same order) or by question
    # We'll match by index first, but also verify questions match
    comparison_data = []

    min_length = min(len(base_results), len(fine_tuned_results))

    for i in range(min_length):
        base_result = base_results[i]
        fine_tuned_result = fine_tuned_results[i]

        # Verify questions match (they should if same dataset)
        base_question = base_result["question"]
        fine_tuned_question = fine_tuned_result["question"]

        if base_question != fine_tuned_question:
            print(f"Warning: Questions don't match at index {i}")
            print(f"  Base: {base_question[:50]}...")
            print(f"  Fine-tuned: {fine_tuned_question[:50]}...")

        human_response = human_responses.get(base_question)
        if human_response is None:
            print(f"Warning: No human response found for question at index {i}")

        comparison_data.append(
            {
                "question": base_question,
                "human_response": human_response if human_response is not None else "",
                "base_response": base_result["response"],
                "fine_tuned_response": fine_tuned_result["response"],
            }
        )

    # Write to CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "human_response",
                "base_response",
                "fine_tuned_response",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_data)

    print(f"CSV file created: {output_csv_path}")
    print(f"Total rows: {len(comparison_data)}")
    return output_csv_path


def load_comparison_csv(csv_path):
    """
    Load comparison CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of dictionaries with question, human_response, base_response, fine_tuned_response
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "question": row["question"],
                    "human_response": row.get("human_response", ""),
                    "base_response": row["base_response"],
                    "fine_tuned_response": row["fine_tuned_response"],
                }
            )
    return data


# Example usage
if __name__ == "__main__":
    eval_results_dir = Path("eval_results")

    base_file = eval_results_dir / "base_first30.json"
    fine_tuned_file = eval_results_dir / "FL_train4_peft25_first30.json"
    output_csv = eval_results_dir / "comparison.csv"

    if base_file.exists() and fine_tuned_file.exists():
        # Create CSV from JSON files and HF dataset
        create_comparison_csv(base_file, fine_tuned_file, output_csv)

        # Example: Load the CSV
        # data = load_comparison_csv(output_csv)
        # print(f"\nLoaded {len(data)} rows from CSV")
    else:
        print("Files not found. Please check:")
        print(f"  Base file: {base_file}")
        print(f"  Fine-tuned file: {fine_tuned_file}")
