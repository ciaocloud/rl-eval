#!/usr/bin/env python3
"""
Math Reasoning Model Evaluation Script

Evaluates LLMs on math reasoning benchmarks using vLLM for efficient inference.
Supports multiple answer formats, templates, and grading methods.

Usage:
    python evaluate.py --model Qwen/Qwen2.5-Math-1.5B --template qwen_math
    python evaluate.py --model /path/to/checkpoint --template r1 --save
"""

import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fire
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model: str = "Qwen/Qwen2.5-Math-1.5B"
    template: str = "qwen_math"  # qwen_math, r1, llama_instruct
    tasks: str = "aime,amc,math,minerva,olympiad_bench"  # can include "gsm8k"
    dataset_path: str = "./understand-r1-zero/datasets/evaluation_suite"
    gsm8k_data_dir: str = None  # Path to preprocessed GSM8K parquet (optional)
    
    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 3000
    n_samples: int = 1
    
    # vLLM parameters
    max_model_len: int = 4096
    batch_size: int = 16
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    
    # Output
    output_dir: str = "./rl-eval/results"
    save: bool = False
    save_responses: bool = False
    max_problems: int = 999999  # Limit for quick testing


# =============================================================================
# Math Grading Functions
# =============================================================================

class MathGrader:
    """
    Math answer grading with multiple verification methods.
    
    Supports:
    - LaTeX boxed answers: \\boxed{answer}
    - R1 answer tags: <answer>answer</answer>
    - Numeric normalization
    - Symbolic equivalence (via sympy)
    """
    
    # Common substitutions for normalization
    SUBSTITUTIONS = [
        ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
        (r"\ ", ""), (" ", ""), ("mbox", "text"),
        (",\\text{and}", ","), ("\\text{and}", ","),
    ]
    
    REMOVED_EXPRESSIONS = [
        "square", "ways", "integers", "dollars", "mph", "inches",
        "hours", "km", "units", "points", "feet", "minutes", "cents",
        "degrees", "cm", "pounds", "meters", "\\text{}", "\\dots",
    ]
    
    @staticmethod
    def extract_boxed(text: str) -> Optional[str]:
        """Extract content from \\boxed{...}."""
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
        if idx < 0:
            return None
        
        i = idx
        num_braces = 0
        right_idx = None
        
        while i < len(text):
            if text[i] == "{":
                num_braces += 1
            elif text[i] == "}":
                num_braces -= 1
                if num_braces == 0:
                    right_idx = i
                    break
            i += 1
        
        if right_idx is None:
            return None
        
        # Extract content between braces
        content = text[idx:right_idx + 1]
        # Remove \\boxed{ and }
        if content.startswith("\\boxed{"):
            content = content[7:-1]
        elif content.startswith("\\fbox{"):
            content = content[6:-1]
        
        return content
    
    @staticmethod
    def extract_answer_tag(text: str) -> Optional[str]:
        """Extract content from <answer>...</answer>."""
        if "<answer>" not in text:
            return None
        
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # If answer contains boxed, extract that too
            if "\\boxed" in answer:
                boxed = MathGrader.extract_boxed(answer)
                if boxed:
                    return boxed
            return answer
        return None
    
    @staticmethod
    def normalize(answer: str) -> str:
        """Normalize an answer string for comparison."""
        if answer is None:
            return ""
        
        answer = str(answer).strip()
        
        # Apply substitutions
        for before, after in MathGrader.SUBSTITUTIONS:
            answer = answer.replace(before, after)
        
        # Remove common expressions
        for expr in MathGrader.REMOVED_EXPRESSIONS:
            answer = answer.replace(expr, "")
        
        # Normalize LaTeX
        answer = re.sub(r"\\text\{(.*?)\}", r"\1", answer)
        answer = re.sub(r"\\textbf\{(.*?)\}", r"\1", answer)
        answer = re.sub(r"\\overline\{(.*?)\}", r"\1", answer)
        
        # Normalize fractions: \frac12 -> \frac{1}{2}
        answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", answer)
        answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", answer)
        
        # Remove dollar signs
        answer = answer.replace("$", "")
        
        # Normalize numbers with commas
        if answer.replace(",", "").isdigit():
            answer = answer.replace(",", "")
        
        # Remove whitespace
        answer = answer.replace(" ", "").lower()
        
        return answer
    
    @staticmethod
    def numeric_equal(pred: str, gt: str, tol: float = 1e-4) -> bool:
        """Check if two values are numerically equal."""
        def try_eval(s):
            """Try to evaluate a string as a number, including fractions."""
            s = s.replace(",", "").strip()
            try:
                return float(s)
            except ValueError:
                pass
            # Try fraction format: a/b
            if "/" in s:
                try:
                    parts = s.split("/")
                    if len(parts) == 2:
                        return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
            return None
        
        try:
            pred_val = try_eval(pred)
            gt_val = try_eval(gt)
            if pred_val is None or gt_val is None:
                return False
            return abs(pred_val - gt_val) <= tol * max(abs(gt_val), 1)
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def symbolic_equal(pred: str, gt: str) -> bool:
        """Check symbolic equivalence using sympy."""
        try:
            from sympy import simplify, sympify, Rational, N
            from sympy.parsing.latex import parse_latex
            
            def try_parse(s):
                """Try multiple parsing methods."""
                s_clean = s.replace("\\\\", "\\").strip()
                
                # Try LaTeX fraction pattern: \frac{a}{b}
                frac_match = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", s_clean)
                if frac_match:
                    try:
                        num = sympify(frac_match.group(1))
                        den = sympify(frac_match.group(2))
                        return num / den
                    except:
                        pass
                
                # Try simple fraction: a/b
                if "/" in s_clean and "\\" not in s_clean:
                    try:
                        parts = s_clean.split("/")
                        if len(parts) == 2:
                            return Rational(parts[0].strip(), parts[1].strip())
                    except:
                        pass
                
                # Try parse_latex
                try:
                    return parse_latex(s_clean)
                except:
                    pass
                
                # Try sympify
                try:
                    return sympify(s_clean)
                except:
                    pass
                
                return None
            
            pred_sym = try_parse(pred)
            gt_sym = try_parse(gt)
            
            if pred_sym is None or gt_sym is None:
                return False
            
            # Try direct equality
            if str(pred_sym) == str(gt_sym):
                return True
            
            # Try simplification
            try:
                if simplify(pred_sym - gt_sym) == 0:
                    return True
            except:
                pass
            
            # Try numeric comparison
            try:
                pred_val = float(N(pred_sym))
                gt_val = float(N(gt_sym))
                if abs(pred_val - gt_val) < 1e-6:
                    return True
            except:
                pass
            
            return False
        except ImportError:
            return False
        except Exception:
            return False
    
    @classmethod
    def grade(cls, pred: str, gt: str, fast: bool = True) -> bool:
        """
        Grade a prediction against ground truth.
        
        Args:
            pred: Model prediction (may contain \\boxed{})
            gt: Ground truth answer
            fast: If False, use slower but more thorough symbolic checking
            
        Returns:
            True if prediction matches ground truth
        """
        if pred is None or gt is None:
            return False
        
        # Normalize both
        pred_norm = cls.normalize(pred)
        gt_norm = cls.normalize(gt)
        
        # Handle ground truth in boxed format
        if "\\boxed" in str(gt):
            gt_extracted = cls.extract_boxed(str(gt))
            if gt_extracted:
                gt_norm = cls.normalize(gt_extracted)
        
        # String equality
        if pred_norm == gt_norm:
            return True
        
        # Numeric equality
        if cls.numeric_equal(pred_norm, gt_norm):
            return True
        
        # Symbolic equality (slower)
        if not fast:
            if cls.symbolic_equal(pred, gt):
                return True
        
        return False


def boxed_reward_fn(response: str, gt: str, fast: bool = True) -> Tuple[dict, float]:
    """
    Grade response expecting \\boxed{answer} format.
    
    Returns:
        (info_dict, reward): info contains 'formatted' bool, reward is 0.0 or 1.0
    """
    extracted = MathGrader.extract_boxed(response)
    
    if extracted is None:
        return {"formatted": False, "extracted": None}, 0.0
    
    # Handle list of acceptable answers
    if isinstance(gt, list):
        is_correct = any(MathGrader.grade(extracted, g, fast) for g in gt)
    else:
        is_correct = MathGrader.grade(extracted, str(gt), fast)
    
    return {"formatted": True, "extracted": extracted}, 1.0 if is_correct else 0.0


def answer_tag_reward_fn(response: str, gt: str, fast: bool = True) -> Tuple[dict, float]:
    """
    Grade response expecting <think>...</think> <answer>...</answer> format.
    
    Returns:
        (info_dict, reward): info contains 'formatted' bool, reward is 0.0 or 1.0
    """
    # Check for proper format
    if "</think>" not in response or "<answer>" not in response:
        return {"formatted": False, "extracted": None}, 0.0
    
    extracted = MathGrader.extract_answer_tag(response)
    
    if extracted is None:
        return {"formatted": True, "extracted": None}, 0.0
    
    # Handle list of acceptable answers
    if isinstance(gt, list):
        is_correct = any(MathGrader.grade(extracted, g, fast) for g in gt)
    else:
        is_correct = MathGrader.grade(extracted, str(gt), fast)
    
    return {"formatted": True, "extracted": extracted}, 1.0 if is_correct else 0.0


# =============================================================================
# Prompt Templates
# =============================================================================

def apply_qwen_math_template(question: str) -> str:
    """Qwen-Math style template with system prompt."""
    return (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_r1_template(question: str) -> str:
    """R1/DeepSeek-style reasoning template."""
    return (
        "A conversation between User and Assistant. The User asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is "
        "enclosed within <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think> <answer> answer here </answer>.\n"
        f"User: {question}\n"
        "Assistant: <think>"
    )


def apply_llama_instruct_template(question: str, tokenizer=None) -> str:
    """Llama-instruct template using chat template."""
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback without tokenizer
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def apply_simple_template(question: str) -> str:
    """
    Simple question-answer template (for backwards compatibility).
    
    This matches the old veRL default format. Use this to evaluate models
    that were trained before proper templating was implemented.
    """
    return f"Question: {question}\n\nAnswer: Let me solve this step by step.\n"


def get_template_and_grader(template: str, model_name: str = None):
    """
    Get the appropriate template function and grader based on template name.
    
    Returns:
        (template_fn, grader_fn, stop_sequences)
    """
    if template == "qwen_math":
        return apply_qwen_math_template, boxed_reward_fn, ["<|im_end|>", "<|endoftext|>"]
    
    elif template == "r1":
        return apply_r1_template, answer_tag_reward_fn, ["</answer>"]
    
    elif template == "llama_instruct":
        # Load tokenizer for proper chat template
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name) if model_name else None
            template_fn = lambda q: apply_llama_instruct_template(q, tokenizer)
        except:
            template_fn = apply_llama_instruct_template
        return template_fn, boxed_reward_fn, ["<|eot_id|>"]
    
    elif template == "simple":
        # For models trained with old veRL simple format
        # Uses boxed grader but expects simpler output
        return apply_simple_template, boxed_reward_fn, ["\n\nQuestion:", "###"]
    
    else:
        raise ValueError(f"Unknown template: {template}. Available: qwen_math, r1, llama_instruct, simple")


# =============================================================================
# Evaluation Engine
# =============================================================================

class Evaluator:
    """Main evaluation engine using vLLM."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.template_fn = None
        self.grader_fn = None
        self.stop_sequences = None
        
    def setup(self):
        """Initialize model and template."""
        import vllm
        
        print(f"\n{'='*60}")
        print(f"Loading model: {self.config.model}")
        print(f"Template: {self.config.template}")
        print(f"Tensor Parallel: {self.config.tensor_parallel_size}")
        print(f"{'='*60}\n")
        
        # Initialize vLLM
        self.model = vllm.LLM(
            self.config.model,
            max_model_len=self.config.max_model_len,
            dtype="bfloat16",
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            trust_remote_code=True,
            enable_prefix_caching=True,
            swap_space=16,
        )
        
        # Get template and grader
        self.template_fn, self.grader_fn, self.stop_sequences = get_template_and_grader(
            self.config.template, self.config.model
        )
        
        print(f"Model loaded successfully!")
        print(f"Stop sequences: {self.stop_sequences}")
    
    def load_datasets(self) -> Dict[str, any]:
        """Load evaluation datasets."""
        from datasets import load_from_disk, load_dataset, Dataset
        
        task_list = [t.strip() for t in self.config.tasks.split(",")]
        result_datasets = {}
        
        # Load GSM8K separately if requested
        if "gsm8k" in task_list:
            gsm8k_data = self._load_gsm8k()
            if gsm8k_data is not None:
                result_datasets["gsm8k"] = gsm8k_data
            task_list = [t for t in task_list if t != "gsm8k"]
        
        # Load other datasets from evaluation suite
        if task_list:
            dataset_path = Path(self.config.dataset_path)
            if not dataset_path.exists():
                # Try relative to script
                dataset_path = Path(__file__).parent.parent / "understand-r1-zero/datasets/evaluation_suite"
            
            if dataset_path.exists():
                print(f"Loading datasets from: {dataset_path}")
                datasets = load_from_disk(str(dataset_path))
                for k, v in datasets.items():
                    if k in task_list:
                        result_datasets[k] = v
            elif task_list:
                print(f"Warning: Dataset path not found: {dataset_path}")
                print(f"Skipping tasks: {task_list}")
        
        return result_datasets
    
    def _load_gsm8k(self):
        """Load GSM8K dataset from local parquet or HuggingFace."""
        from datasets import load_dataset, Dataset
        
        # Try local parquet first
        if self.config.gsm8k_data_dir:
            parquet_path = Path(self.config.gsm8k_data_dir) / "test.parquet"
            if parquet_path.exists():
                print(f"Loading GSM8K from local parquet: {parquet_path}")
                raw_data = Dataset.from_parquet(str(parquet_path))
                # Convert to expected format (problem, answer)
                def convert_verl_format(example):
                    # Handle verl preprocessed format
                    if "prompt" in example and isinstance(example["prompt"], list):
                        question = example["prompt"][0]["content"]
                    elif "extra_info" in example and "question" in example["extra_info"]:
                        question = example["extra_info"]["question"]
                    else:
                        question = str(example.get("prompt", ""))
                    
                    # Get ground truth
                    if "reward_model" in example and "ground_truth" in example["reward_model"]:
                        answer = example["reward_model"]["ground_truth"]
                    else:
                        answer = str(example.get("answer", ""))
                    
                    return {"problem": question, "answer": answer}
                
                return raw_data.map(convert_verl_format)
        
        # Fall back to HuggingFace
        print("Loading GSM8K from HuggingFace (openai/gsm8k)...")
        try:
            gsm8k = load_dataset("openai/gsm8k", "main", split="test")
            # Convert to expected format (problem, answer)
            def convert_hf_format(example):
                return {
                    "problem": example["question"],
                    "answer": self._extract_gsm8k_answer(example["answer"]),
                }
            return gsm8k.map(convert_hf_format)
        except Exception as e:
            print(f"Warning: Could not load GSM8K: {e}")
            return None
    
    @staticmethod
    def _extract_gsm8k_answer(answer_str: str) -> str:
        """Extract final answer from GSM8K answer format (#### ANSWER)."""
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_str)
        if match:
            return match.group(1).replace(",", "")
        return answer_str
    
    def evaluate_task(self, task_name: str, dataset) -> Dict:
        """Evaluate on a single task."""
        import vllm
        
        # Prepare prompts
        problems = dataset["problem"][:self.config.max_problems]
        answers = dataset["answer"][:self.config.max_problems]
        
        prompts = [self.template_fn(p) for p in problems]
        
        # Create sampling params
        sampling_params = vllm.SamplingParams(
            n=self.config.n_samples,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stop=self.stop_sequences,
        )
        
        # Add stop token in output for answer tag format
        if self.config.template == "r1":
            sampling_params.include_stop_str_in_output = True
        
        print(f"\n[{task_name}] Evaluating {len(prompts)} problems...")
        
        # Generate
        outputs = self.model.generate(prompts, sampling_params)
        
        # Grade
        scores = []
        lengths = []
        formatted_count = 0
        details = []
        
        for i, output in enumerate(outputs):
            gt = answers[i]
            sample_scores = []
            sample_lengths = []
            
            for j, completion in enumerate(output.outputs):
                response = completion.text
                info, reward = self.grader_fn(response, gt, fast=False)
                
                sample_scores.append(reward)
                sample_lengths.append(len(completion.token_ids))
                
                if info.get("formatted", False):
                    formatted_count += 1
                
                if j == 0:  # Store first sample details
                    detail = {
                        "problem": problems[i],
                        "gt": gt,
                        "extracted": info.get("extracted"),
                        "correct": reward > 0,
                        "formatted": info.get("formatted", False),
                    }
                    if self.config.save_responses:
                        detail["response"] = response
                    details.append(detail)
            
            # For pass@k, use max score across samples
            scores.append(max(sample_scores))
            lengths.extend(sample_lengths)
        
        accuracy = np.mean(scores)
        avg_length = np.mean(lengths)
        format_rate = formatted_count / (len(outputs) * self.config.n_samples)
        
        print(f"[{task_name}] {len(problems)}/{len(problems)}: "
              f"{accuracy*100:.1f}% ({int(sum(scores))}/{len(scores)})")
        
        return {
            "accuracy": accuracy,
            "total": len(scores),
            "correct": int(sum(scores)),
            "avg_length": avg_length,
            "format_rate": format_rate,
            "details": details if self.config.save else None,
        }
    
    def run(self) -> Dict:
        """Run full evaluation."""
        self.setup()
        datasets = self.load_datasets()
        
        results = {}
        all_details = {}
        
        for task_name, dataset in datasets.items():
            task_result = self.evaluate_task(task_name, dataset)
            results[task_name] = {
                "accuracy": task_result["accuracy"],
                "total": task_result["total"],
                "correct": task_result["correct"],
                "avg_length": task_result["avg_length"],
                "format_rate": task_result["format_rate"],
            }
            if task_result["details"]:
                all_details[task_name] = task_result["details"]
        
        # Compute averages
        avg_accuracy = np.mean([r["accuracy"] for r in results.values()])
        avg_length = np.mean([r["avg_length"] for r in results.values()])
        avg_format = np.mean([r["format_rate"] for r in results.values()])
        
        # Print summary
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {self.config.model}")
        print(f"Template: {self.config.template}")
        print(f"Average Score: {avg_accuracy*100:.1f}%\n")
        
        print("Benchmark Scores:")
        for task, res in results.items():
            print(f"  {task}: {res['accuracy']*100:.1f}%")
        
        print(f"\nAverage Response Length: {avg_length:,.0f} tokens")
        print(f"Format Compliance: {avg_format*100:.1f}%")
        print(f"{'='*80}\n")
        
        # Prepare output
        output = {
            "model": self.config.model,
            "template": self.config.template,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "n_samples": self.config.n_samples,
            },
            "results": results,
            "average": avg_accuracy,
            "avg_length": avg_length,
            "format_rate": avg_format,
        }
        
        # Save results
        if self.config.save:
            self._save_results(output, all_details)
        
        return output
    
    def _save_results(self, output: Dict, details: Dict):
        """Save results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from model name
        model_name = self.config.model.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = output_dir / f"{model_name}_{timestamp}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved summary: {summary_file}")
        
        # Save details
        if details:
            details_file = output_dir / f"{model_name}_{timestamp}_details.json"
            with open(details_file, "w") as f:
                json.dump(details, f, indent=2)
            print(f"Saved details: {details_file}")


# =============================================================================
# CLI
# =============================================================================

def main(
    model: str = "Qwen/Qwen2.5-Math-1.5B",
    template: str = "qwen_math",
    tasks: str = "aime,amc,math,minerva,olympiad_bench",
    dataset_path: str = None,
    gsm8k_data_dir: str = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 3000,
    n_samples: int = 1,
    max_model_len: int = 4096,
    batch_size: int = 16,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    output_dir: str = "./rl-eval/results",
    save: bool = False,
    save_responses: bool = False,
    max_problems: int = 999999,
):
    """
    Evaluate math reasoning models.
    
    Args:
        model: Model name or path (HuggingFace or local)
        template: Prompt template (qwen_math, r1, llama_instruct)
        tasks: Comma-separated list of tasks (aime,amc,math,minerva,olympiad_bench,gsm8k)
        dataset_path: Path to evaluation suite dataset
        gsm8k_data_dir: Path to preprocessed GSM8K parquet files (optional)
        temperature: Sampling temperature (0 for greedy)
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        n_samples: Number of samples per problem (for pass@k)
        max_model_len: Maximum model context length
        batch_size: Inference batch size
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization
        output_dir: Directory to save results
        save: Whether to save detailed results
        save_responses: Whether to save full model responses
        max_problems: Maximum problems per task (for quick testing)
    
    Examples:
        # Basic evaluation
        python evaluate.py --model Qwen/Qwen2.5-Math-1.5B --template qwen_math
        
        # With R1 template
        python evaluate.py --model your_model --template r1 --save
        
        # Quick test on MATH only
        python evaluate.py --model your_model --tasks math --max_problems 50
        
        # GSM8K evaluation
        python evaluate.py --model your_model --tasks gsm8k --template qwen_math
        
        # GSM8K with local preprocessed data
        python evaluate.py --model your_model --tasks gsm8k --gsm8k_data_dir /workspace/data/gsm8k
    """
    # Find dataset path (only needed for non-GSM8K tasks)
    task_list = [t.strip() for t in tasks.split(",")]
    non_gsm8k_tasks = [t for t in task_list if t != "gsm8k"]
    
    if dataset_path is None and non_gsm8k_tasks:
        # Try common locations
        candidates = [
            "./understand-r1-zero/datasets/evaluation_suite",
            "../understand-r1-zero/datasets/evaluation_suite",
            str(Path(__file__).parent.parent / "understand-r1-zero/datasets/evaluation_suite"),
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                dataset_path = candidate
                break
        else:
            if non_gsm8k_tasks:
                print(f"Warning: Could not find evaluation dataset for tasks: {non_gsm8k_tasks}")
                print("Specify --dataset_path or only use --tasks gsm8k")
            dataset_path = "./understand-r1-zero/datasets/evaluation_suite"
    
    config = EvalConfig(
        model=model,
        template=template,
        tasks=tasks,
        dataset_path=dataset_path or "./understand-r1-zero/datasets/evaluation_suite",
        gsm8k_data_dir=gsm8k_data_dir,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_samples=n_samples,
        max_model_len=max_model_len,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        output_dir=output_dir,
        save=save,
        save_responses=save_responses,
        max_problems=max_problems,
    )
    
    evaluator = Evaluator(config)
    results = evaluator.run()
    
    return results


if __name__ == "__main__":
    fire.Fire(main)

