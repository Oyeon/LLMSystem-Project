import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import pandas as pd
import os
import argparse

def main():
    # Set up argument parser
    # parser = argparse.ArgumentParser(description="Run DeepSeek-V2-Lite inference on StrategyQA dataset")
    # parser.add_argument("--cache-dir", type=str, default=None, 
    #                     help="Custom directory for Hugging Face cache (default: use HF default)")
    # parser.add_argument("--batch-size", type=int, default=1, 
    #                     help="Batch size for inference (default: 1)")
    # parser.add_argument("--quantization", type=str, default="4bit", choices=["none", "4bit", "8bit"],
    #                     help="Quantization method (default: 4bit)")
    # args = parser.parse_args()
    batch_size = 1
    quantization = ["none", "4bit", "8bit"][0]
    cache_dir = os.getcwd() 
    if cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        print(f"Using custom Hugging Face cache directory: {cache_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")

    # Load DeepSeek-V2-Lite model and tokenizer
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer with trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Configure quantization for memory efficiency
    
    if quantization == "4bit":
        print("Using 4-bit quantization for memory efficiency")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif quantization == "8bit":
        print("Using 8-bit quantization for memory efficiency")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        print("Using fp16 precision (no quantization)")
        bnb_config = None
    
    # Load model with memory optimizations
    model_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    print(f"Model loaded successfully with {'quantization' if bnb_config else 'fp16 precision'}")
    
    # Load StrategyQA dataset
    dataset_name = "allenai/strategyqa"
    print(f"Loading dataset: {dataset_name}")
    
    # Load validation split
    dataset = load_dataset(
        dataset_name, 
        split="validation",
        cache_dir=cache_dir
    )
    
    # For testing with limited resources, you can use a subset
    # dataset = dataset.select(range(min(100, len(dataset))))
    
    # Set inference parameters
    
    all_predictions = []
    all_reasoning = []
    
    # Optimize CUDA memory usage
    torch.cuda.empty_cache()
    
    # Process the dataset in batches with tqdm to show progress
    for i in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        
        # Prepare prompts for strategic QA
        prompts = []
        for question in batch["question"]:
            prompt = f"""Answer the following question that requires multi-step reasoning. Think through the problem step by step before providing your answer.

Question: {question}

To solve this question, I'll break it down into steps:
1."""
            prompts.append(prompt)
        
        # Set smaller max input length to save memory
        tokenizer.model_max_length = 1024
        
        # Tokenize inputs with padding and truncation
        inputs = tokenizer(
            prompts, 
            padding=True, 
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate completions with optimized parameters
        with torch.no_grad():
            # Use gradient checkpointing for memory efficiency
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                
            # Clear CUDA cache between batches
            if i > 0:
                torch.cuda.empty_cache()
                
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced to save memory
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode the generated output
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        batch_predictions = []
        batch_reasoning = []
        for text in generated_texts:
            # Extract the reasoning and answer
            response = text.split("Question: ")[1].strip() if "Question: " in text else text
            
            # Extract binary answer (yes/no) - look at the end of the response
            answer_text = response.lower().split("\n")[-1]
            if "yes" in answer_text and "no" not in answer_text[:answer_text.find("yes")]:
                prediction = True
            elif "no" in answer_text:
                prediction = False
            else:
                # Default guess if no clear yes/no
                prediction = None
            
            # Extract reasoning
            reasoning = response
            
            batch_predictions.append(prediction)
            batch_reasoning.append(reasoning)
        
        all_predictions.extend(batch_predictions)
        all_reasoning.extend(batch_reasoning)
        
        # Force garbage collection to free memory
        if i % 10 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    # Filter out None predictions (where model didn't produce a clear yes/no)
    valid_indices = [i for i, pred in enumerate(all_predictions) if pred is not None]
    valid_predictions = [all_predictions[i] for i in valid_indices]
    valid_ground_truth = [dataset[i]["answer"] for i in valid_indices]
    valid_questions = [dataset[i]["question"] for i in valid_indices]
    valid_reasoning = [all_reasoning[i] for i in valid_indices]
    
    # Calculate accuracy
    correct = sum(p == gt for p, gt in zip(valid_predictions, valid_ground_truth))
    accuracy = correct / len(valid_predictions) if valid_predictions else 0
    
    print(f"\nInference complete!")
    print(f"Valid predictions: {len(valid_predictions)}/{len(dataset)}")
    print(f"Accuracy on valid predictions: {accuracy:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        "question": valid_questions,
        "prediction": valid_predictions,
        "ground_truth": valid_ground_truth,
        "correct": [p == gt for p, gt in zip(valid_predictions, valid_ground_truth)],
        "reasoning": valid_reasoning
    })
    
    results_df.to_csv("strategyqa_results.csv", index=False)
    print(f"Results saved to strategyqa_results.csv")

if __name__ == "__main__":
    main()