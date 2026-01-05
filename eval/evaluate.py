#!/usr/bin/env python3
"""
EscherVerse Benchmark Evaluation Script

A unified evaluation script for testing Vision-Language Models on the Escher-Bench benchmark.

Usage:
    # Local models (Transformers)
    python evaluate.py --model qwen3-vl-8b --data_path ./Escher-Bench.json --video_dir ./videos
    
    # API models
    python evaluate.py --model gpt-4o --api_key YOUR_KEY --data_path ./Escher-Bench.json

Supported Models:
    Local: qwen3-vl-8b, qwen3-vl-4b, qwen2.5-vl-7b, llava-onevision-7b, internvl3-8b
    API: gpt-4o, gpt-4o-mini, gemini-2.5-pro, gemini-2.5-flash, claude-3.5-sonnet
"""

import json
import os
import re
import base64
import time
import argparse
import difflib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
import io
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_MODELS = {
    # Local models (Transformers-based)
    "qwen3-vl-8b": {"type": "local", "model_path": "Qwen/Qwen3-VL-8B-Instruct", "class": "qwen3"},
    "qwen3-vl-4b": {"type": "local", "model_path": "Qwen/Qwen3-VL-4B-Instruct", "class": "qwen3"},
    "qwen3-vl-2b": {"type": "local", "model_path": "Qwen/Qwen3-VL-2B-Instruct", "class": "qwen3"},
    "qwen2.5-vl-7b": {"type": "local", "model_path": "Qwen/Qwen2.5-VL-7B-Instruct", "class": "qwen2.5"},
    "qwen2.5-vl-3b": {"type": "local", "model_path": "Qwen/Qwen2.5-VL-3B-Instruct", "class": "qwen2.5"},
    "llava-onevision-7b": {"type": "local", "model_path": "lmms-lab/llava-onevision-qwen2-7b-ov", "class": "llava"},
    "internvl3-8b": {"type": "local", "model_path": "OpenGVLab/InternVL3-8B", "class": "internvl"},
    
    # API models
    "gpt-4o": {"type": "api", "model_name": "gpt-4o"},
    "gpt-4o-mini": {"type": "api", "model_name": "gpt-4o-mini"},
    "gemini-2.5-pro": {"type": "api", "model_name": "gemini-2.5-pro"},
    "gemini-2.5-flash": {"type": "api", "model_name": "gemini-2.5-flash"},
    "claude-3.5-sonnet": {"type": "api", "model_name": "claude-3-5-sonnet-20241022"},
}

DEFAULT_NUM_FRAMES = 16
DEFAULT_MAX_TOKENS = 4096
SIMILARITY_THRESHOLD = 0.75

# =============================================================================
# Video Processing
# =============================================================================

def extract_video_frames(video_path: Path, num_frames: int = 16) -> List[Image.Image]:
    """Extract frames uniformly from a video file."""
    if not video_path.exists():
        print(f"  [Error] Video file not found: {video_path}")
        return []
    
    try:
        import av
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if total_frames == 0:
            duration = float(stream.duration * stream.time_base)
            fps = float(stream.average_rate)
            total_frames = int(duration * fps)
        
        if total_frames < 1:
            container.close()
            return []
        
        num_frames = min(total_frames, num_frames)
        indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())
        
        pil_images = []
        frame_idx = 0
        for frame in container.decode(video=0):
            if frame_idx in indices:
                pil_images.append(frame.to_image())
                if len(pil_images) >= num_frames:
                    break
            frame_idx += 1
        
        container.close()
        return pil_images
    except ImportError:
        # Fallback to OpenCV
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return []
        
        num_frames = min(total_frames, num_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        pil_images = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                pil_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        cap.release()
        return pil_images
    except Exception as e:
        print(f"  [Error] Failed to process video {video_path}: {e}")
        return []


def pil_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# =============================================================================
# Prompt and Answer Processing
# =============================================================================

def create_prompt(question: str) -> str:
    """Create standardized evaluation prompt."""
    return f"""Carefully examine the provided video frames and answer the following question.
Your final answer must be enclosed exclusively between `<answer>` and `</answer>` tags.

Question: {question}"""


def extract_answer_from_tags(text: str) -> str:
    """Extract content between <answer> tags."""
    if not text:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def extract_option_letters(text: str, question_type: str) -> str:
    """Extract option letters from answer text."""
    text = text.strip()
    
    if question_type == "Multiple-Select":
        matches = re.findall(r'\b([A-D])\)?', text, re.IGNORECASE)
        letters = sorted(set(m.upper() for m in matches))
        return ','.join(letters) if letters else text
    else:
        match = re.search(r'\b([A-D])\)?', text, re.IGNORECASE)
        return match.group(1).upper() if match else text


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return ' '.join(text.lower().strip().split())


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate combined text similarity."""
    text1, text2 = text1.lower().strip(), text2.lower().strip()
    if not text1 or not text2:
        return 0.0
    
    # Token overlap
    tokens1, tokens2 = set(text1.split()), set(text2.split())
    token_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2) if tokens1 | tokens2 else 0.0
    
    # Sequence similarity
    seq_sim = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Character n-gram overlap
    def get_ngrams(text, n=3):
        return set(text[i:i+n] for i in range(len(text)-n+1))
    ngrams1, ngrams2 = get_ngrams(text1), get_ngrams(text2)
    char_sim = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2) if ngrams1 | ngrams2 else 0.0
    
    return 0.5 * token_sim + 0.3 * seq_sim + 0.2 * char_sim


def evaluate_answer(prediction: str, ground_truth: str, question_type: str) -> Dict:
    """Evaluate answer correctness based on question type."""
    pred_clean = prediction.strip()
    gt_clean = ground_truth.strip()
    
    if question_type == "Single-Choice":
        pred_opt = extract_option_letters(pred_clean, question_type)
        gt_opt = extract_option_letters(gt_clean, question_type)
        is_correct = pred_opt.upper() == gt_opt.upper()
        return {"is_correct": is_correct, "score": 1.0 if is_correct else 0.0,
                "prediction_clean": pred_opt, "ground_truth_clean": gt_opt, "eval_method": "hard_match"}
    
    elif question_type == "Multiple-Select":
        pred_opt = extract_option_letters(pred_clean, question_type)
        gt_opt = extract_option_letters(gt_clean, question_type)
        is_correct = pred_opt == gt_opt
        return {"is_correct": is_correct, "score": 1.0 if is_correct else 0.0,
                "prediction_clean": pred_opt, "ground_truth_clean": gt_opt, "eval_method": "hard_match"}
    
    elif question_type == "True/False":
        pred_norm = normalize_text(pred_clean)
        gt_norm = normalize_text(gt_clean)
        pred_true = 'true' in pred_norm
        pred_false = 'false' in pred_norm
        gt_true = 'true' in gt_norm
        gt_false = 'false' in gt_norm
        is_correct = (pred_true and gt_true) or (pred_false and gt_false)
        return {"is_correct": is_correct, "score": 1.0 if is_correct else 0.0,
                "prediction_clean": "True" if pred_true else "False",
                "ground_truth_clean": "True" if gt_true else "False", "eval_method": "hard_match"}
    
    elif question_type == "Fill-in-the-Blank":
        pred_norm = normalize_text(pred_clean)
        gt_norm = normalize_text(gt_clean)
        
        if pred_norm == gt_norm:
            return {"is_correct": True, "score": 1.0, "prediction_clean": pred_clean,
                    "ground_truth_clean": gt_clean, "eval_method": "hard_match", "similarity": 1.0}
        
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return {"is_correct": True, "score": 1.0, "prediction_clean": pred_clean,
                    "ground_truth_clean": gt_clean, "eval_method": "substring_match", "similarity": 1.0}
        
        similarity = calculate_similarity(pred_clean, gt_clean)
        score = 0.5 if similarity >= SIMILARITY_THRESHOLD else 0.0
        return {"is_correct": False, "score": score, "prediction_clean": pred_clean,
                "ground_truth_clean": gt_clean, "eval_method": "similarity", "similarity": similarity}
    
    # Default
    is_correct = normalize_text(pred_clean) == normalize_text(gt_clean)
    return {"is_correct": is_correct, "score": 1.0 if is_correct else 0.0,
            "prediction_clean": pred_clean, "ground_truth_clean": gt_clean, "eval_method": "hard_match"}


# =============================================================================
# Model Inference
# =============================================================================

class ModelInference:
    """Unified model inference interface."""
    
    def __init__(self, model_key: str, api_key: str = None, api_base: str = None):
        self.model_key = model_key
        self.config = SUPPORTED_MODELS.get(model_key)
        if not self.config:
            raise ValueError(f"Unsupported model: {model_key}")
        
        self.model_type = self.config["type"]
        self.model = None
        self.processor = None
        self.client = None
        
        if self.model_type == "api":
            self._init_api_client(api_key, api_base)
        else:
            self._init_local_model()
    
    def _init_api_client(self, api_key: str, api_base: str):
        """Initialize OpenAI-compatible API client."""
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=api_base or "https://api.openai.com/v1"
        )
        self.model_name = self.config["model_name"]
    
    def _init_local_model(self):
        """Initialize local transformers model."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        model_path = self.config["model_path"]
        model_class = self.config["class"]
        
        print(f"Loading model: {model_path}")
        
        if model_class in ["qwen3", "qwen2.5"]:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        elif model_class == "llava":
            from transformers import LlavaOnevisionForConditionalGeneration
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        elif model_class == "internvl":
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, low_cpu_mem_usage=True
            ).eval()
            self.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"Model loaded successfully")
    
    def generate(self, prompt: str, images: List[Image.Image]) -> Optional[str]:
        """Generate response from model."""
        if self.model_type == "api":
            return self._generate_api(prompt, images)
        else:
            return self._generate_local(prompt, images)
    
    def _generate_api(self, prompt: str, images: List[Image.Image], max_retries: int = 3) -> Optional[str]:
        """Generate using API."""
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(img)}"}
            })
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=0.0,
                    timeout=120
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  [API Error] Attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
        return None
    
    def _generate_local(self, prompt: str, images: List[Image.Image]) -> Optional[str]:
        """Generate using local model."""
        model_class = self.config["class"]
        
        try:
            if model_class in ["qwen3", "qwen2.5"]:
                return self._generate_qwen(prompt, images)
            elif model_class == "llava":
                return self._generate_llava(prompt, images)
            elif model_class == "internvl":
                return self._generate_internvl(prompt, images)
        except Exception as e:
            print(f"  [Inference Error]: {e}")
            return None
    
    def _generate_qwen(self, prompt: str, images: List[Image.Image]) -> str:
        """Generate using Qwen model."""
        content = [{"type": "text", "text": prompt}]
        for _ in images:
            content.append({"type": "image"})
        
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=DEFAULT_MAX_TOKENS, do_sample=False)
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def _generate_llava(self, prompt: str, images: List[Image.Image]) -> str:
        """Generate using LLaVA model."""
        import torch
        conversation = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt_text, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=DEFAULT_MAX_TOKENS, do_sample=False)
        
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def _generate_internvl(self, prompt: str, images: List[Image.Image]) -> str:
        """Generate using InternVL model."""
        import torch
        pixel_values_list = []
        num_patches_list = []
        
        for img in images:
            pixel_values = self.model.load_image(img, max_num=12).to(self.model.device, dtype=self.model.dtype)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
        
        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_tokens = ''.join([f'<image-{i+1}>: <image>\n' for i in range(len(images))])
        question = f'{image_tokens}{prompt}'
        
        response = self.model.chat(self.processor, pixel_values, question, 
                                   dict(max_new_tokens=DEFAULT_MAX_TOKENS, do_sample=False),
                                   num_patches_list=num_patches_list)
        return response


# =============================================================================
# Statistics and Results
# =============================================================================

def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate evaluation statistics."""
    stats = {
        "total": len(results),
        "correct": 0,
        "partial_correct": 0,
        "incorrect": 0,
        "error": 0,
        "total_score": 0.0,
        "accuracy": 0.0,
        "by_category": defaultdict(lambda: {"total": 0, "score": 0.0, "accuracy": 0.0}),
        "by_scene_type": defaultdict(lambda: {"total": 0, "score": 0.0, "accuracy": 0.0}),
        "by_question_type": defaultdict(lambda: {"total": 0, "score": 0.0, "accuracy": 0.0})
    }
    
    for item in results:
        category = item.get("category", "Unknown")
        scene_type = item.get("scene_type", "Unknown")
        question_type = item.get("question_type", "Unknown")
        score = item.get("score", 0.0)
        
        stats["total_score"] += score
        
        if item.get("is_correct") is None:
            stats["error"] += 1
        elif score == 1.0:
            stats["correct"] += 1
        elif score > 0.0:
            stats["partial_correct"] += 1
        else:
            stats["incorrect"] += 1
        
        for dim, key in [("by_category", category), ("by_scene_type", scene_type), ("by_question_type", question_type)]:
            stats[dim][key]["total"] += 1
            stats[dim][key]["score"] += score
    
    if stats["total"] > 0:
        stats["accuracy"] = stats["total_score"] / stats["total"]
    
    for dim in ["by_category", "by_scene_type", "by_question_type"]:
        for key, data in stats[dim].items():
            if data["total"] > 0:
                data["accuracy"] = data["score"] / data["total"]
        stats[dim] = dict(stats[dim])
    
    return stats


def save_results(results: List[Dict], stats: Dict, output_path: Path, model_name: str):
    """Save evaluation results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_items": len(results),
            "num_frames": DEFAULT_NUM_FRAMES
        },
        "statistics": stats,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Evaluation Results - {model_name}")
    print("=" * 70)
    print(f"Overall Accuracy: {stats['accuracy']:.2%} ({stats['total_score']:.1f}/{stats['total']})")
    print(f"Correct: {stats['correct']}, Partial: {stats['partial_correct']}, Incorrect: {stats['incorrect']}, Error: {stats['error']}")
    print("\nBy Question Type:")
    for qtype, data in sorted(stats['by_question_type'].items()):
        print(f"  {qtype}: {data['accuracy']:.2%} ({data['score']:.1f}/{data['total']})")
    print("\nBy Category:")
    for cat, data in sorted(stats['by_category'].items()):
        print(f"  {cat}: {data['accuracy']:.2%}")
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def run_evaluation(args):
    """Run benchmark evaluation."""
    # Load benchmark data
    print(f"Loading benchmark data from: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    print(f"Loaded {len(benchmark_data)} evaluation items")
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    model = ModelInference(args.model, api_key=args.api_key, api_base=args.api_base)
    
    # Run evaluation
    results = []
    video_dir = Path(args.video_dir)
    
    for i, item in enumerate(tqdm(benchmark_data, desc="Evaluating")):
        video_filename = item.get("P")
        if not video_filename:
            continue
        
        video_path = video_dir / video_filename
        images = extract_video_frames(video_path, args.num_frames)
        if not images:
            continue
        
        question = item.get('Q', '')
        question_type = item.get('question_type', 'Unknown')
        ground_truth = item.get('A', '')
        category = item.get('C', 'Unknown')
        scene_type = item.get('scene_type', 'Unknown')
        
        prompt = create_prompt(question)
        response = model.generate(prompt, images)
        
        if response:
            prediction = extract_answer_from_tags(response)
            eval_result = evaluate_answer(prediction, ground_truth, question_type)
        else:
            prediction = "[ERROR]"
            eval_result = {"is_correct": None, "score": 0.0, "prediction_clean": "[ERROR]",
                          "ground_truth_clean": ground_truth, "eval_method": "error"}
        
        result_item = {
            "index": item.get('index', i),
            "video": video_filename,
            "question": question,
            "question_type": question_type,
            "category": category,
            "scene_type": scene_type,
            "ground_truth": ground_truth,
            "model_prediction": prediction,
            "is_correct": eval_result["is_correct"],
            "score": eval_result["score"],
            **{k: v for k, v in eval_result.items() if k not in ["is_correct", "score"]}
        }
        results.append(result_item)
        
        # Periodic save
        if (i + 1) % 100 == 0:
            temp_stats = calculate_statistics(results)
            print(f"\n  Progress: {i+1}/{len(benchmark_data)}, Current Accuracy: {temp_stats['accuracy']:.2%}")
    
    # Final statistics and save
    final_stats = calculate_statistics(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"results_{args.model}_{timestamp}.json"
    save_results(results, final_stats, output_path, args.model)


def main():
    parser = argparse.ArgumentParser(description="EscherVerse Benchmark Evaluation")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=list(SUPPORTED_MODELS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--data_path", "-d", type=str, required=True,
                        help="Path to Escher-Bench.json")
    parser.add_argument("--video_dir", "-v", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output_dir", "-o", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--num_frames", "-n", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of frames to extract per video")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for API models")
    parser.add_argument("--api_base", type=str, default=None,
                        help="API base URL for API models")
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
