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
import hashlib
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
try:
    from tqdm import tqdm
except ImportError:  # Keep scoring/parser-only use lightweight.
    def tqdm(iterable, **_kwargs):
        return iterable

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_MODELS = {
    # Local models (Transformers-based)
    "qwen3-vl-8b": {"type": "local", "model_path": "Qwen/Qwen3-VL-8B-Instruct", "class": "qwen3"},
    "qwen3-vl-8b-thinking": {"type": "local", "model_path": "Qwen/Qwen3-VL-8B-Thinking", "class": "qwen3"},
    "qwen3-vl-4b-thinking": {"type": "local", "model_path": "Qwen/Qwen3-VL-4B-Thinking", "class": "qwen3"},
    "qwen3-vl-2b-instruct": {"type": "local", "model_path": "Qwen/Qwen3-VL-2B-Instruct", "class": "qwen3"},
    "qwen3-vl-4b": {"type": "local", "model_path": "Qwen/Qwen3-VL-4B-Instruct", "class": "qwen3"},
    "qwen3-vl-2b": {"type": "local", "model_path": "Qwen/Qwen3-VL-2B-Instruct", "class": "qwen3"},
    "qwen2.5-vl-7b": {"type": "local", "model_path": "Qwen/Qwen2.5-VL-7B-Instruct", "class": "qwen2.5"},
    "qwen2.5-vl-32b-awq": {"type": "local", "model_path": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", "class": "qwen2.5"},
    "qwen2.5-vl-3b": {"type": "local", "model_path": "Qwen/Qwen2.5-VL-3B-Instruct", "class": "qwen2.5"},
    "llava-onevision-7b": {"type": "local", "model_path": "lmms-lab/llava-onevision-qwen2-7b-ov", "class": "llava"},
    "internvl3-8b": {"type": "local", "model_path": "OpenGVLab/InternVL3-8B", "class": "internvl"},
    "internvl3-2b": {"type": "local", "model_path": "OpenGVLab/InternVL3-2B", "class": "internvl"},
    
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
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    nonempty = [match.strip() for match in matches if match.strip()]
    return "\n".join(nonempty) if nonempty else text.strip()


def extract_option_letters(text: str, question_type: str) -> str:
    """Extract option letters from answer text."""
    text = text.strip()

    # Accept explicit option labels while avoiding ordinary words such as
    # "answer", "between", "correct", and the article "a".
    label_matches = re.findall(
        r"(?<![A-Za-z0-9])([A-D])"
        r"(?=(?:[ \t]*(?:[).,:;/]|$|\band\b|&)|[ \t]*\r?\n))",
        text,
    )
    answer_phrase_matches = re.findall(
        r"\b(?:answer|option|choice)s?\s*(?:is|are|:)?\s*([A-D])\b",
        text,
        re.IGNORECASE,
    )
    letters = sorted(
        set(label_matches + [letter.upper() for letter in answer_phrase_matches])
    )
    if question_type == "Multiple-Select":
        return ",".join(letters) if letters else text
    return letters[0] if len(letters) == 1 else (",".join(letters) or text)


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
    
    def __init__(
        self,
        model_key: str,
        api_key: str = None,
        api_base: str = None,
        model_path: str = None,
        adapter_path: str = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        awq_preserve_lm_head: bool = False,
    ):
        self.model_key = model_key
        self.config = SUPPORTED_MODELS.get(model_key)
        if not self.config:
            raise ValueError(f"Unsupported model: {model_key}")
        
        self.model_type = self.config["type"]
        self.model = None
        self.processor = None
        self.client = None
        self.model_path_override = model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.awq_preserve_lm_head = awq_preserve_lm_head
        self.last_generation_metadata = {}
        
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
        from transformers import AutoProcessor
        
        model_path = self.model_path_override or self.config["model_path"]
        model_class = self.config["class"]
        
        print(f"Loading model: {model_path}")
        
        if model_class == "qwen3":
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            if self.adapter_path:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
        elif model_class == "qwen2.5":
            from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

            model_kwargs = {}
            model_dtype = (
                torch.float16
                if self.model_key.endswith("-awq")
                else torch.bfloat16
            )
            if self.awq_preserve_lm_head:
                model_config = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
                quantization_config = dict(model_config.quantization_config)
                preserved = list(
                    quantization_config.get("modules_to_not_convert", [])
                )
                if "lm_head" not in preserved:
                    preserved.append("lm_head")
                quantization_config["modules_to_not_convert"] = preserved
                model_config.quantization_config = quantization_config
                model_kwargs["config"] = model_config
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                device_map="auto",
                trust_remote_code=True,
                **model_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
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
            self.processor = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
        
        print(f"Model loaded successfully")
    
    def generate(self, prompt: str, images: List[Image.Image]) -> Optional[str]:
        """Generate response from model."""
        self.last_generation_metadata = {}
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
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    timeout=120
                )
                usage = getattr(response, "usage", None)
                self.last_generation_metadata = {
                    "finish_reason": response.choices[0].finish_reason,
                    "generated_token_count": getattr(usage, "completion_tokens", None),
                    "hit_max_tokens": response.choices[0].finish_reason == "length",
                }
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
        
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_tokens, do_sample=False
        )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        token_ids = generated_ids[0].tolist()
        eos_token_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids = set(eos_token_ids or [])
        eos_observed = any(token_id in eos_token_ids for token_id in token_ids)
        hit_max_tokens = len(token_ids) >= self.max_tokens and not eos_observed
        self.last_generation_metadata = {
            "generated_token_count": len(token_ids),
            "eos_observed": eos_observed,
            "hit_max_tokens": hit_max_tokens,
            "finish_reason": (
                "max_tokens" if hit_max_tokens else "eos" if eos_observed else "unknown"
            ),
        }
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def _generate_llava(self, prompt: str, images: List[Image.Image]) -> str:
        """Generate using LLaVA model."""
        import torch
        conversation = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt_text, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False
            )
        
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def _generate_internvl(self, prompt: str, images: List[Image.Image]) -> str:
        """Generate using InternVL model."""
        import torch
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda image: image.convert("RGB")
                    if image.mode != "RGB"
                    else image
                ),
                transforms.Resize(
                    (448, 448), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        pixel_values = torch.stack([transform(image) for image in images]).to(
            self.model.device, dtype=self.model.dtype
        )
        num_patches_list = [1] * len(images)
        video_prefix = "".join(
            f"Frame{index + 1}: <image>\n" for index in range(len(images))
        )
        question = f"{video_prefix}{prompt}"

        response = self.model.chat(
            self.processor,
            pixel_values,
            question,
            dict(max_new_tokens=self.max_tokens, do_sample=False),
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        response_token_count = len(
            self.processor.encode(response, add_special_tokens=False)
        )
        self.last_generation_metadata = {
            "generated_token_count": response_token_count,
            "hit_max_tokens": response_token_count >= self.max_tokens,
            "finish_reason": (
                "max_tokens_estimate"
                if response_token_count >= self.max_tokens
                else "length_below_limit"
            ),
            "finish_reason_auditable": False,
        }
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


def save_results(
    results: List[Dict],
    stats: Dict,
    output_path: Path,
    model_name: str,
    model_path: str = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    run_metadata: Dict = None,
):
    """Save evaluation results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_items": len(results),
            "num_frames": num_frames,
            "model_path": model_path,
            "max_tokens": max_tokens,
            **(run_metadata or {}),
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
    model = ModelInference(
        args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        awq_preserve_lm_head=args.awq_preserve_lm_head,
    )
    
    # Run evaluation
    results = []
    video_dir = Path(args.video_dir)
    
    for i, item in enumerate(tqdm(benchmark_data, desc="Evaluating")):
        video_filename = item.get("P")
        question = item.get('Q', '')
        question_type = item.get('question_type', 'Unknown')
        ground_truth = item.get('A', '')
        category = item.get('C', 'Unknown')
        scene_type = item.get('scene_type', 'Unknown')

        base_result = {
            "index": item.get('index', i),
            "video": video_filename,
            "question": question,
            "question_type": question_type,
            "category": category,
            "scene_type": scene_type,
            "target_capability": item.get("target_capability"),
            "ground_truth": ground_truth,
        }
        if not video_filename:
            results.append({
                **base_result,
                "raw_response": "",
                "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
                "answer_tag_count": 0,
                "generation_metadata": {},
                "model_prediction": "[ERROR]",
                "is_correct": None,
                "score": 0.0,
                "prediction_clean": "[ERROR]",
                "ground_truth_clean": ground_truth,
                "eval_method": "error",
                "error_type": "missing_video_filename",
            })
            continue

        video_path = video_dir / video_filename
        images = extract_video_frames(video_path, args.num_frames)
        if not images:
            results.append({
                **base_result,
                "raw_response": "",
                "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
                "answer_tag_count": 0,
                "generation_metadata": {},
                "model_prediction": "[ERROR]",
                "is_correct": None,
                "score": 0.0,
                "prediction_clean": "[ERROR]",
                "ground_truth_clean": ground_truth,
                "eval_method": "error",
                "error_type": "video_frame_extraction_failed",
            })
            continue
        
        prompt = create_prompt(question)
        response = model.generate(prompt, images)
        
        if response:
            prediction = extract_answer_from_tags(response)
            eval_result = evaluate_answer(prediction, ground_truth, question_type)
        else:
            prediction = "[ERROR]"
            eval_result = {"is_correct": None, "score": 0.0, "prediction_clean": "[ERROR]",
                          "ground_truth_clean": ground_truth, "eval_method": "error"}
        
        raw_response = response or ""
        result_item = {
            **base_result,
            "raw_response": raw_response,
            "raw_response_sha256": hashlib.sha256(
                raw_response.encode("utf-8")
            ).hexdigest(),
            "answer_tag_count": len(
                re.findall(r"<answer>.*?</answer>", raw_response, re.DOTALL | re.IGNORECASE)
            ),
            "generation_metadata": dict(model.last_generation_metadata),
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
    result_indices = [item["index"] for item in results]
    run_integrity = {
        "input_items": len(benchmark_data),
        "output_items": len(results),
        "unique_output_indices": len(set(result_indices)),
        "inference_errors": final_stats["error"],
        "complete": (
            len(results) == len(benchmark_data)
            and len(set(result_indices)) == len(benchmark_data)
            and final_stats["error"] == 0
        ),
    }
    evaluator_path = Path(__file__).resolve()
    data_path = Path(args.data_path).resolve()
    run_metadata = {
        "evaluator_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "video_dir": str(video_dir.resolve()),
        "run_integrity": run_integrity,
        "awq_preserve_lm_head": args.awq_preserve_lm_head,
        "adapter_path": args.adapter_path,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"results_{args.model}_{timestamp}.json"
    save_results(
        results,
        final_stats,
        output_path,
        args.model,
        args.model_path,
        args.max_tokens,
        args.num_frames,
        run_metadata,
    )
    if not run_integrity["complete"]:
        raise RuntimeError(f"Evaluation integrity check failed: {run_integrity}")


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
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override the configured local checkpoint path")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional PEFT adapter to load on the local base model")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Maximum generated tokens per response")
    parser.add_argument(
        "--awq_preserve_lm_head",
        action="store_true",
        help=(
            "Treat lm_head as unquantized when an AWQ checkpoint stores "
            "lm_head.weight but omits it from modules_to_not_convert"
        ),
    )
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
