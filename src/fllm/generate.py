checkpoint_root = "./FL_iid_qwen0.5b_newcluster_seed75"
SEED = 75
import os
import sys
import argparse
import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import from fllm package (works both as script and module)
try:
	from fllm.fed import cfg, get_tokenizer_and_data_collator, get_train_test_indices, CLUSTERED_CSV_PATH
except ImportError:
	# Fallback for running as script from src/fllm directory
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from fllm.fed import cfg, get_tokenizer_and_data_collator, get_train_test_indices, CLUSTERED_CSV_PATH

def get_latest_checkpoint(checkpoint_root):
	# If directory does not exist or is not a directory, signal that no checkpoints are available
	if not checkpoint_root or not os.path.isdir(checkpoint_root):
		return None
	checkpoints = [d for d in os.listdir(checkpoint_root) if d.startswith('peft_') or d.startswith('checkpoint-')]
	if not checkpoints:
		return None
	def extract_num(name):
		if name.startswith('peft_'):
			return int(name.split('peft_')[-1])
		elif name.startswith('checkpoint-'):
			return int(name.split('checkpoint-')[-1])
		return -1
	checkpoints = sorted(checkpoints, key=extract_num)
	latest_checkpoint = os.path.join(checkpoint_root, checkpoints[-1])
	# Check if adapter_model subdirectory exists (for PEFT checkpoints)
	adapter_model_path = os.path.join(latest_checkpoint, "adapter_model")
	if os.path.exists(adapter_model_path):
		return adapter_model_path
	return latest_checkpoint

def load_model_and_tokenizer(checkpoint_path, model_name=None):
	model_name = model_name or cfg.model.name
	tokenizer, _ = get_tokenizer_and_data_collator(
		model_name,
		cfg.train,
		cfg.model.use_fast_tokenizer,
		cfg.train.padding_side,
	)
	# Llama-specific: add special tokens if needed
	if 'llama' in model_name.lower() or 'LlamaTokenizer' in str(type(tokenizer)):
		print('Adding special tokens for LLaMA.')
		special_tokens = {}
		if getattr(tokenizer, 'eos_token', None) is None:
			special_tokens['eos_token'] = '</s>'
		if getattr(tokenizer, 'bos_token', None) is None:
			special_tokens['bos_token'] = '<s>'
		if getattr(tokenizer, 'unk_token', None) is None:
			special_tokens['unk_token'] = '<unk>'
		if special_tokens:
			tokenizer.add_special_tokens(special_tokens)
	print(f"Loaded tokenizer from {model_name} with vocab size {len(tokenizer)}")
	base_model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		low_cpu_mem_usage=True,
		device_map="auto" if torch.cuda.is_available() else None,
	)
	if checkpoint_path is None:
		# No checkpoint specified: use base model only
		print("Using base model only (no PEFT)" + (f" – model: {model_name}" if model_name != cfg.model.name else ""))
		model = base_model
	else:
		# Checkpoint specified: use fine-tuned (base + PEFT) model
		print(f"Using latest checkpoint: {checkpoint_path}")
		# Resize base model embeddings to match training tokenizer size (handles added special tokens)
		checkpoint_vocab_size = len(tokenizer)
		current_model_size = base_model.get_input_embeddings().num_embeddings
		if current_model_size != checkpoint_vocab_size:
			print(f"Resizing model embeddings from {current_model_size} to {checkpoint_vocab_size} to match training tokenizer.")
			base_model.resize_token_embeddings(checkpoint_vocab_size)

		model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	# Set model to eval mode to disable dropout and ensure deterministic behavior
	model.eval()
	# Disable gradient computation globally
	torch.set_grad_enabled(False)
	return model, tokenizer, device

def generate_response(prompt, model, tokenizer, device, max_new_tokens=512):
	if hasattr(tokenizer, "apply_chat_template") and isinstance(prompt, list):
		prompt_str = tokenizer.apply_chat_template(
			prompt,
			tokenize=False,
			add_generation_prompt=True
		)
	else:
		prompt_str = prompt if isinstance(prompt, str) else str(prompt)
	inputs = tokenizer(prompt_str, return_tensors='pt').to(device)
	with torch.no_grad():
		# Set temperature=0 for deterministic generation
		outputs = model.generate(
			**inputs, 
			max_new_tokens=max_new_tokens, 
			eos_token_id=tokenizer.eos_token_id,
			temperature=0.0,
			do_sample=False  # Greedy decoding for deterministic output
		)
	generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
	return tokenizer.decode(generated_ids, skip_special_tokens=True)

def main():
	parser = argparse.ArgumentParser(description="Generate responses from a model (optionally with PEFT checkpoint).")
	parser.add_argument("--model", type=str, default=cfg.model.name, help="Hugging Face model name (e.g. microsoft/phi-2). If omitted, uses cfg.model.name.")
	parser.add_argument("--checkpoint_root", type=str, default=checkpoint_root, help="Directory containing PEFT checkpoints. If provided and a checkpoint exists, the fine-tuned model (base+PEFT) is used; otherwise the base model is used.")
	parser.add_argument("--output_dir", type=str, default="./generated_output6", help="Directory to save generated JSON")
	parser.add_argument("--output_name", type=str, default=f"{checkpoint_root}.json", help="Output filename (default: derived from checkpoint or model)")
	parser.add_argument("--test_size", type=float, default=0.02, help="Test set fraction (default: 0.02)")
	parser.add_argument("--seed", type=int, default=SEED, help="Random seed for test split (default: SEED)")
	parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate (default: 512)")
	args = parser.parse_args()

	output_dir = args.output_dir
	os.makedirs(output_dir, exist_ok=True)

	if args.checkpoint_root:
		checkpoint_path = get_latest_checkpoint(args.checkpoint_root)
		if checkpoint_path is None:
			# No checkpoints found: fall back to base model
			print(f"No checkpoints found in {args.checkpoint_root}, falling back to base model.")
			checkpoint_path = None
		model_name = args.model if args.model else None
		model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, model_name=model_name)
		if checkpoint_path is not None:
			output_name_local = args.output_name or os.path.basename(args.checkpoint_root.rstrip("/")) + ".json"
		else:
			output_name_local = args.output_name or (model_name or cfg.model.name).replace("/", "_") + ".json"
	else:
		# No checkpoint_root specified: always use base model
		model_name = args.model if args.model else None
		model, tokenizer, device = load_model_and_tokenizer(None, model_name=model_name)
		output_name_local = args.output_name or (model_name or cfg.model.name).replace("/", "_") + ".json"

	output_json = os.path.join(output_dir, output_name_local)
	# Ensure model is in eval mode (already set in load_model_and_tokenizer, but double-check)
	model.eval()

	print("Loading test data from CSV (same split method as fed.py)...")
	if not os.path.isfile(CLUSTERED_CSV_PATH):
		raise FileNotFoundError(
			f"Clustered dataset CSV not found at {CLUSTERED_CSV_PATH}. "
			"Run clustering.py to create clustered_dataset.csv first."
		)
	df = pd.read_csv(CLUSTERED_CSV_PATH)
	ds_full = Dataset.from_pandas(df, preserve_index=False)
	n_total = len(ds_full)
	_, test_indices = get_train_test_indices(n_total, args.test_size, args.seed)
	test_data = ds_full.select(test_indices.tolist())

	batch_size = 10
	total = len(test_data)
	# Collect indices of examples that have a valid question (same as before)
	indices_to_run = []
	for idx, example in enumerate(test_data):
		user_question = example.get("input") or example.get("instruction") or example.get("Context")
		if user_question:
			indices_to_run.append(idx)
	total_valid = len(indices_to_run)

	file_opened = False
	for batch_start in range(0, total_valid, batch_size):
		batch_indices = indices_to_run[batch_start : batch_start + batch_size]
		batch_results = []
		for idx in batch_indices:
			example = test_data[idx]
			user_question = example.get("input") or example.get("instruction") or example.get("Context")
			prompt = [{"role": "user", "content": user_question}]
			response = generate_response(prompt, model, tokenizer, device, max_new_tokens=args.max_new_tokens)
			batch_results.append({"question": user_question, "response": response})
			print(f"[{batch_start + len(batch_results)}/{total_valid}] Q: {user_question}\nA: {response}\n{'-'*40}")

		# Append this batch to the same file (valid JSON array)
		chunk = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in batch_results)
		is_first_batch = not file_opened
		is_last_batch = (batch_start + len(batch_indices)) >= total_valid

		if is_first_batch:
			with open(output_json, "w") as f:
				f.write("[\n")
				f.write(chunk)
				f.write("\n")
			file_opened = True
		else:
			with open(output_json, "a") as f:
				f.write(",\n")
				f.write(chunk)
				f.write("\n")
		if is_last_batch:
			with open(output_json, "a") as f:
				f.write("]")
		print(f"Saved batch of {len(batch_results)} examples to {output_json} (total so far: {batch_start + len(batch_results)})")

	print(f"Done. Saved {total_valid} question-response pairs to {output_json}")

if __name__ == "__main__":
	main()
