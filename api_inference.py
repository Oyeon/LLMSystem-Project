import torch
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitsandbytes.nn import Linear8bitLt
from tqdm import tqdm
import requests
import os
from typing import List, Dict, Any, Tuple

# Set custom cache directories for Hugging Face
os.environ["HF_HOME"] = os.getcwd() 
os.environ["HF_DATASETS_CACHE"] = os.getcwd() 
CACHE_DIR = os.getcwd() 

# Environment variables for API (replace with your actual values)
API_URL = os.environ.get("LLAMA_API_URL", "https://api.together.xyz/v1/completions")
API_KEY = os.environ.get("LLAMA_API_KEY", "your_api_key_here")

class StrategyQAEvaluator:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """
        Initialize the StrategyQA evaluator
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.dataset = self._load_dataset()
        self.results = {"correct": 0, "total": 0, "accuracy": 0.0, "examples": []}
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load StrategyQA dataset"""
        dataset = load_dataset(
            "metaeval/strategy-qa", 
            split="validation",
            cache_dir=os.environ["HF_DATASETS_CACHE"]
        )
        # Convert to list of dictionaries for easier processing
        return [{"question": item["question"], "answer": item["answer"]} for item in dataset]
    
    def _create_prompt(self, question: str) -> str:
        """
        Create a prompt for the StrategyQA question
        
        Args:
            question: The question to answer
            
        Returns:
            Formatted prompt
        """
        return f"""Answer the following question with 'yes' or 'no'. Before giving your answer, explain your reasoning step-by-step.

Question: {question}

Reasoning:"""
    
    def call_api(self, prompt: str) -> str:
        """
        Call the LLaMA API to get a response
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Model response
        """
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "meta-llama/Llama-2-7b-hf-8bit",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["\n\n"]
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            print(f"API call failed: {e}")
            return ""

    def extract_answer(self, response: str) -> str:
        """
        Extract yes/no answer from the model's response
        
        Args:
            response: The model's response
            
        Returns:
            'yes' or 'no' based on the response
        """
        response_lower = response.lower()
        
        # Check for explicit yes/no at the end
        if "answer: yes" in response_lower:
            return "yes"
        elif "answer: no" in response_lower:
            return "no"
        
        # Count occurrences of yes and no
        yes_count = response_lower.count("yes")
        no_count = response_lower.count("no")
        
        # If one is clearly more frequent
        if yes_count > no_count + 1:
            return "yes"
        elif no_count > yes_count + 1:
            return "no"
        
        # Default case - check the last sentence
        last_sentences = response_lower.split(".")[-3:]
        for sentence in reversed(last_sentences):
            if "yes" in sentence and "no" not in sentence:
                return "yes"
            elif "no" in sentence and "yes" not in sentence:
                return "no"
        
        # If all else fails, return the most frequent
        return "yes" if yes_count >= no_count else "no"
    
    def evaluate(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate the model on StrategyQA
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        # Limit to num_samples
        samples = self.dataset[:num_samples]
        
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            question = sample["question"]
            true_answer = "yes" if sample["answer"] else "no"
            
            # Create prompt and get API response
            prompt = self._create_prompt(question)
            response = self.call_api(prompt)
            predicted_answer = self.extract_answer(response)
            
            # Check if correct
            is_correct = predicted_answer == true_answer
            if is_correct:
                self.results["correct"] += 1
            self.results["total"] += 1
            
            # Save example
            self.results["examples"].append({
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response
            })
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(samples)}, Current Accuracy: {self.results['correct']/(i+1):.4f}")
        
        # Calculate final accuracy
        self.results["accuracy"] = self.results["correct"] / self.results["total"]
        return self.results
    
    def save_results(self, output_file: str = "strategyqa_results.json") -> None:
        """
        Save evaluation results to file
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")

def run_local_quantized_test(num_samples: int = 10):
    """
    Run a small test using locally loaded 8-bit quantized model
    """
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        cache_dir=CACHE_DIR
    )
    
    # Load model in 8-bit
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    
    # Load dataset
    dataset = load_dataset(
        "metaeval/strategy-qa", 
        split="validation",
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    samples = dataset[:num_samples]
    
    results = []
    
    for sample in tqdm(samples, desc="Testing local model"):
        question = sample["question"]
        true_answer = "yes" if sample["answer"] else "no"
        
        # Create prompt
        prompt = f"Answer the following question with 'yes' or 'no'. Before giving your answer, explain your reasoning step-by-step.\n\nQuestion: {question}\n\nReasoning:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9
            )
            
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        
        # Extract answer
        response_lower = response.lower()
        if "yes" in response_lower and "no" not in response_lower:
            predicted = "yes"
        elif "no" in response_lower and "yes" not in response_lower:
            predicted = "no"
        else:
            predicted = "yes" if response_lower.count("yes") > response_lower.count("no") else "no"
            
        # Check if correct
        is_correct = predicted == true_answer
        
        results.append({
            "question": question,
            "true_answer": true_answer,
            "predicted": predicted,
            "is_correct": is_correct
        })
        
    # Calculate accuracy
    accuracy = sum(r["is_correct"] for r in results) / len(results)
    print(f"Local test accuracy: {accuracy:.4f}")
    return results

def main():
    """Main function"""
    print("StrategyQA Evaluation with 8-bit quantized LLaMA-2-7B")
    print(f"Using custom cache directories:")
    print(f"- HF_HOME: {os.environ['HF_HOME']}")
    print(f"- HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"- Models cache: {CACHE_DIR}")
    
    # Ensure cache directories exist
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check for API key
    if API_KEY == "your_api_key_here":
        print("API key not set. Falling back to local test.")
        run_local_quantized_test(num_samples=10)
        return
    
    # Number of samples to evaluate
    num_samples = 100  # Adjust based on your API rate limits
    
    # Initialize evaluator
    evaluator = StrategyQAEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate(num_samples)
    
    # Print results
    print(f"\nFinal Results:")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    # Save results
    evaluator.save_results()

if __name__ == "__main__":
    main()