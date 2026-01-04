#!/usr/bin/env python3
"""
GSM8K Evaluation Script using verl methods and vLLM for fast inference.

Prerequisites:
    1. Preprocess GSM8K dataset:
       python scripts/preprocess_gsm8k.py --local_save_dir data/gsm8k

Usage:
    # Evaluate with vLLM
    python rl-eval/eval_gsm8k_verl.py --model Qwen/Qwen2.5-0.5B-Instruct
    
    # pass@8 evaluation with sampling
    python rl-eval/eval_gsm8k_verl.py --model Qwen/Qwen2.5-0.5B-Instruct --num_samples 8 --temperature 0.7
    
    # Use flexible answer extraction
    python rl-eval/eval_gsm8k_verl.py --model Qwen/Qwen2.5-0.5B-Instruct --extraction_method flexible
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.special import comb
from tqdm import tqdm
import datasets

from vllm import LLM, SamplingParams

# Import verl's reward scoring
from verl.utils.reward_score import gsm8k as verl_gsm8k
from verl.utils.reward_score import default_compute_score


# =============================================================================
# Metrics
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k (from Codex paper).
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def majority_vote(answers: List[str]) -> Optional[str]:
    """Get the most common answer from a list."""
    if not answers:
        return None
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None
    return Counter(valid_answers).most_common(1)[0][0]


# =============================================================================
# Reward Computation (following verl's NaiveRewardManager pattern)
# =============================================================================

def compute_rewards_batch(
    responses: List[str],
    ground_truths: List[str],
    data_sources: List[str],
    extraction_method: str = "strict",
) -> List[float]:
    """
    Compute rewards for a batch of responses using verl's default_compute_score.
    
    This follows the pattern from verl/workers/reward_manager/naive.py
    
    Args:
        responses: List of model response strings
        ground_truths: List of ground truth answers
        data_sources: List of data source identifiers (e.g., "openai/gsm8k")
        extraction_method: "strict" or "flexible"
    
    Returns:
        List of reward scores
    """
    rewards = []
    
    for response, ground_truth, data_source in zip(responses, ground_truths, data_sources):
        # Use verl's default_compute_score which routes based on data_source
        score = default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
        )
        
        if isinstance(score, dict):
            reward = score.get("score", 0.0)
        else:
            reward = float(score)
        
        rewards.append(reward)
    
    return rewards


def extract_answers_batch(
    responses: List[str],
    extraction_method: str = "strict",
) -> List[Optional[str]]:
    """
    Extract answers from a batch of responses using verl's gsm8k.extract_solution.
    
    Args:
        responses: List of model response strings
        extraction_method: "strict" or "flexible"
    
    Returns:
        List of extracted answers (None if extraction failed)
    """
    return [
        verl_gsm8k.extract_solution(r, method=extraction_method) 
        for r in responses
    ]


# =============================================================================
# vLLM Generator
# =============================================================================

class VLLMGenerator:
    """Fast generation using vLLM."""
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        self.model_name = model_name
        
        print(f"Loading vLLM model: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"Model loaded with vLLM (tp={tensor_parallel_size})")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_samples: int = 1,
    ) -> List[List[str]]:
        """
        Generate responses for a batch of prompts.
        
        Returns:
            List of lists, each containing num_samples responses per prompt.
        """
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=num_samples,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            responses = [o.text for o in output.outputs]
            results.append(responses)
        
        return results


# =============================================================================
# Evaluator
# =============================================================================

class GSM8KEvaluator:
    """Evaluator for GSM8K using verl methods and vLLM."""
    
    def __init__(
        self,
        model_name: str,
        extraction_method: str = "strict",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        self.model_name = model_name
        self.extraction_method = extraction_method
        
        self.generator = VLLMGenerator(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.tokenizer = self.generator.tokenizer
    
    def format_prompt(self, prompt_messages: List[Dict]) -> str:
        """Format prompt using chat template."""
        return self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def evaluate_batch(
        self,
        problems: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of problems.
        
        Uses verl's reward computation pattern:
        1. Generate responses with vLLM
        2. Compute rewards using default_compute_score (routes by data_source)
        3. Extract answers using gsm8k.extract_solution
        """
        # Format prompts
        prompts = [self.format_prompt(p["prompt"]) for p in problems]
        ground_truths = [p["reward_model"]["ground_truth"] for p in problems]
        data_sources = [p.get("data_source", "openai/gsm8k") for p in problems]
        
        # Generate responses with vLLM
        all_responses = self.generator.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
        )
        
        # Process results
        results = []
        for i, (problem, responses) in enumerate(zip(problems, all_responses)):
            ground_truth = ground_truths[i]
            data_source = data_sources[i]
            
            # Compute rewards for all samples using verl's default_compute_score
            scores = compute_rewards_batch(
                responses=responses,
                ground_truths=[ground_truth] * len(responses),
                data_sources=[data_source] * len(responses),
                extraction_method=self.extraction_method,
            )
            
            # Extract answers using verl's gsm8k.extract_solution
            extracted_answers = extract_answers_batch(
                responses=responses,
                extraction_method=self.extraction_method,
            )
            
            n_correct = sum(scores)
            
            result = {
                "index": problem.get("extra_info", {}).get("index", i),
                "question": problem.get("extra_info", {}).get("question", ""),
                "ground_truth": ground_truth,
                "data_source": data_source,
                "responses": responses,
                "extracted_answers": extracted_answers,
                "scores": scores,
                "pass@1": scores[0] if scores else 0,
                "n_correct": n_correct,
                "n_total": len(scores),
            }
            
            # Compute pass@k for various k values
            for k in [1, 4, 8, 16, 32, 64]:
                if k <= num_samples:
                    result[f"pass@{k}"] = pass_at_k(num_samples, int(n_correct), k)
            
            # Compute majority vote accuracy
            if num_samples > 1:
                majority_answer = majority_vote(extracted_answers)
                try:
                    result["maj_vote_correct"] = (
                        float(majority_answer) == float(ground_truth) 
                        if majority_answer else False
                    )
                except (ValueError, TypeError):
                    result["maj_vote_correct"] = (
                        majority_answer == ground_truth 
                        if majority_answer else False
                    )
                result["majority_answer"] = majority_answer
            
            results.append(result)
        
        return results


# =============================================================================
# Data Loading
# =============================================================================

def load_gsm8k_parquet(data_dir: str, split: str = "test", max_problems: int = None) -> List[Dict]:
    """
    Load preprocessed GSM8K from local parquet files.
    
    Args:
        data_dir: Directory containing train.parquet and test.parquet
        split: "train" or "test"
        max_problems: Maximum number of problems to load
    """
    parquet_path = os.path.join(data_dir, f"{split}.parquet")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {parquet_path}. "
            f"Run: python scripts/preprocess_gsm8k.py --local_save_dir {data_dir}"
        )
    
    print(f"Loading {parquet_path}...")
    dataset = datasets.Dataset.from_parquet(parquet_path)
    
    problems = []
    for i, item in enumerate(dataset):
        if max_problems and i >= max_problems:
            break
        problems.append(dict(item))
    
    print(f"Loaded {len(problems)} problems from {split} split")
    return problems


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(
    model_name: str,
    data_dir: str = "data/gsm8k",
    split: str = "test",
    max_problems: int = None,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 512,
    extraction_method: str = "strict",
    batch_size: int = 32,
    output_file: str = None,
    save_responses: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
) -> Dict[str, float]:
    """
    Run evaluation on GSM8K using vLLM and verl's reward functions.
    
    Args:
        model_name: HuggingFace model name or path
        data_dir: Directory containing preprocessed parquet files
        split: Dataset split ("test" or "train")
        max_problems: Maximum number of problems to evaluate
        num_samples: Number of samples per problem (1 for greedy/pass@1)
        temperature: Sampling temperature (0 for greedy)
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum tokens to generate
        extraction_method: "strict" (#### only) or "flexible" (last number)
        batch_size: Batch size for vLLM
        output_file: Path to save results JSON
        save_responses: Whether to save full responses
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model context length
    """
    # Load dataset
    problems = load_gsm8k_parquet(data_dir, split=split, max_problems=max_problems)
    
    # Initialize evaluator
    evaluator = GSM8KEvaluator(
        model_name=model_name,
        extraction_method=extraction_method,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    
    # Evaluate
    print(f"\nEvaluating {len(problems)} problems...")
    print(f"  num_samples: {num_samples}")
    print(f"  temperature: {temperature}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  extraction_method: {extraction_method}")
    print(f"  batch_size: {batch_size}")
    print()
    
    results = []
    all_pass1 = []
    all_pass_k = {k: [] for k in [1, 4, 8, 16, 32, 64] if k <= num_samples}
    all_maj_correct = []
    
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(problems), batch_size), desc="Evaluating"):
        batch = problems[i:i + batch_size]
        batch_results = evaluator.evaluate_batch(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
        )
        
        for result in batch_results:
            all_pass1.append(result["pass@1"])
            for k in all_pass_k:
                if f"pass@{k}" in result:
                    all_pass_k[k].append(result[f"pass@{k}"])
            if "maj_vote_correct" in result:
                all_maj_correct.append(result["maj_vote_correct"])
            
            if not save_responses:
                result.pop("responses", None)
            
            results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Compute final metrics
    metrics = {
        "model": model_name,
        "split": split,
        "n_problems": len(problems),
        "num_samples": num_samples,
        "temperature": temperature,
        "extraction_method": extraction_method,
        "pass@1": np.mean(all_pass1),
        "elapsed_time": elapsed_time,
        "problems_per_second": len(problems) / elapsed_time,
    }
    
    for k, values in all_pass_k.items():
        if values:
            metrics[f"pass@{k}"] = np.mean(values)
    
    if all_maj_correct:
        metrics[f"maj@{num_samples}"] = np.mean(all_maj_correct)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: GSM8K {split} ({len(problems)} problems)")
    print(f"Extraction: {extraction_method}")
    print(f"Time: {elapsed_time:.1f}s ({metrics['problems_per_second']:.2f} problems/s)")
    print()
    print("Metrics:")
    print(f"  pass@1: {metrics['pass@1']*100:.2f}%")
    for k in sorted(all_pass_k.keys()):
        if k > 1 and f"pass@{k}" in metrics:
            print(f"  pass@{k}: {metrics[f'pass@{k}']*100:.2f}%")
    if f"maj@{num_samples}" in metrics:
        print(f"  maj@{num_samples}: {metrics[f'maj@{num_samples}']*100:.2f}%")
    print("=" * 60)
    
    # Save results
    if output_file:
        output_data = {
            "metrics": metrics,
            "config": {
                "model_name": model_name,
                "data_dir": data_dir,
                "split": split,
                "max_problems": max_problems,
                "num_samples": num_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "extraction_method": extraction_method,
                "batch_size": batch_size,
                "tensor_parallel_size": tensor_parallel_size,
            },
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K using verl + vLLM")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model name or path")
    
    # Dataset
    parser.add_argument("--data_dir", type=str, default="/workspace/data/gsm8k",
                        help="Directory containing preprocessed parquet files")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Maximum number of problems to evaluate")
    
    # Generation
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per problem (1 for pass@1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for vLLM")
    
    # Evaluation
    parser.add_argument("--extraction_method", type=str, default="strict",
                        choices=["strict", "flexible"],
                        help="Answer extraction: 'strict' (#### only) or 'flexible' (last number)")
    
    # vLLM options
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model context length")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--save_responses", action="store_true",
                        help="Save full responses in output")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(
        model_name=args.model,
        data_dir=args.data_dir,
        split=args.split,
        max_problems=args.max_problems,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        extraction_method=args.extraction_method,
        batch_size=args.batch_size,
        output_file=args.output,
        save_responses=args.save_responses,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
