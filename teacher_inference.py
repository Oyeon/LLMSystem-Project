import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configuration
model_name = "mistralai/Mistral-7B-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_samples = 500  # Limit number of samples to evaluate, set to None for full dataset
output_file = "mistral_strategyqa_results.json"

print(f"Running on device: {device}")

# Set up 8-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# Load model and tokenizer
print("Loading model and tokenizer with 8-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)

# Load StrategyQA dataset
print("Loading StrategyQA dataset...")
dataset = load_dataset("metaeval/strategy-qa", split="validation")
if max_samples:
    dataset = dataset.select(range(min(max_samples, len(dataset))))

# Format prompts for multi-hop reasoning
def format_prompt(question):
    return f"""Question: {question}
Please think step by step and provide your reasoning before answering the question with 'Yes' or 'No'.
Answer:"""

# Function to extract yes/no answer from model output
def extract_answer(text):
    text = text.lower().strip()
    
    # Look for specific answers at the end of the text
    if text.endswith("answer: yes") or text.endswith("answer: true") or text.endswith("the answer is yes"):
        return "yes"
    elif text.endswith("answer: no") or text.endswith("answer: false") or text.endswith("the answer is no"):
        return "no"
    
    # Extract last sentence if needed
    sentences = text.split(".")
    last_sentence = sentences[-1].strip() if sentences else text
    
    # Check for yes/no in the response
    if "yes" in last_sentence and "no" not in last_sentence:
        return "yes"
    elif "no" in last_sentence and "yes" not in last_sentence:
        return "no"
    
    # Count yes/no occurrences in the full text as a fallback
    yes_count = text.count("yes")
    no_count = text.count("no")
    
    if yes_count > no_count:
        return "yes"
    elif no_count > yes_count:
        return "no"
    
    # If still unclear, return None
    return None

# Evaluate model on dataset
print("Starting evaluation...")
results = []

for i, item in enumerate(tqdm(dataset)):
    question = item["question"]
    true_answer = "yes" if item["answer"] else "no"
    
    # Format the prompt
    prompt = format_prompt(question)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_response = full_response[len(prompt):].strip()
    
    # Extract yes/no answer
    model_answer = extract_answer(model_response)
    
    # Store results
    result = {
        "question": question,
        "true_answer": true_answer,
        "model_response": model_response,
        "extracted_answer": model_answer,
        "correct": model_answer == true_answer if model_answer else False
    }
    results.append(result)

# Calculate metrics
predictions = [result["extracted_answer"] for result in results]
true_labels = ["yes" if item["answer"] else "no" for item in dataset]

# Filter out None predictions for metric calculation
valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
valid_predictions = [predictions[i] for i in valid_indices]
valid_true_labels = [true_labels[i] for i in valid_indices]

if valid_predictions:
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_true_labels, 
        valid_predictions, 
        average='binary',
        pos_label='yes'
    )
else:
    accuracy = precision = recall = f1 = 0.0

# Calculate percentage of unanswered questions
unanswered = predictions.count(None)
unanswered_percent = unanswered / len(predictions) * 100 if predictions else 0

# Summary metrics
summary = {
    "model": model_name,
    "dataset": "StrategyQA",
    "samples_evaluated": len(dataset),
    "quantization": "8-bit",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "unanswered_questions": unanswered,
    "unanswered_percent": unanswered_percent
}

# Save results
with open(output_file, "w") as f:
    json.dump({
        "summary": summary,
        "results": results
    }, f, indent=2)

print(f"\nEvaluation complete. Results saved to {output_file}")
print(f"Summary metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Unanswered questions: {unanswered} ({unanswered_percent:.2f}%)")