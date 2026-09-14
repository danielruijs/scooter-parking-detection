import gc
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

if not torch.cuda.is_available():
    raise SystemError("A CUDA-capable GPU is required to run VLM models.")

DEVICE = "cuda"


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("HF_TOKEN") and "=" in line:
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    if token:
                        os.environ["HF_TOKEN"] = token
                        break
    return token


def extract_prediction(raw_text: str) -> tuple[bool | None, str, bool]:
    """Extract is_proper (bool) and feedback (str) from model output.

    Returns (is_proper, feedback, parse_success).
    """
    cleaned = raw_text.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    target_str = json_match.group(1) if json_match else ""
    if not target_str:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            target_str = cleaned[start_idx : end_idx + 1]

    if target_str:
        try:
            data = json.loads(target_str)
            is_proper = data.get("is_proper")
            feedback = str(data.get("feedback", "")).strip()
            if isinstance(is_proper, bool):
                return is_proper, feedback, True
            if str(is_proper).lower() in ["true", "false"]:
                return str(is_proper).lower() == "true", feedback, True
            status_val = str(
                data.get("status")
                or data.get("parking_status")
                or data.get("assessment_status")
                or ""
            ).lower()
            if "improper" in status_val:
                return False, feedback or str(data), True
            elif "proper" in status_val:
                return True, feedback or str(data), True
        except Exception:
            pass

    # Fallback regex extraction
    proper_match = re.search(r'"is_proper"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    feedback_match = re.search(r'"feedback"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    if proper_match:
        is_proper = proper_match.group(1).lower() == "true"
        feedback = (
            feedback_match.group(1).replace('\\"', '"').strip()
            if feedback_match
            else cleaned
        )
        return is_proper, feedback, True

    return None, cleaned, False


def format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        val_b = num_params / 1_000_000_000
        rounded = round(val_b)
        return f"{rounded}B" if abs(val_b - rounded) < 0.08 else f"{val_b:.1f}B"
    elif num_params >= 1_000_000:
        val_m = num_params / 1_000_000
        if val_m >= 750:
            return f"{num_params / 1_000_000_000:.1f}B"
        return f"{round(val_m / 10) * 10}M" if val_m > 100 else f"{round(val_m)}M"
    return str(num_params)


def count_model_parameters(model: Any) -> int:
    """Count total parameters, accounting for 4-bit packed tensors in bitsandbytes."""
    if model is None:
        return 0
    total = 0
    for p in model.parameters():
        if hasattr(p, "quant_state") or type(p).__name__ == "Params4bit":
            total += p.numel() * 2
        else:
            total += p.numel()
    return total


class VLMAdapter(ABC):
    """Base interface for all vision-language model adapters."""

    def __init__(
        self,
        name: str,
        hf_id: str,
        display_name: str,
        use_4bit: bool,
        release_date: str,
    ):
        self.name = name
        self.hf_id = hf_id
        self.display_name = display_name
        self.use_4bit = use_4bit
        self.release_date = release_date
        self.device = DEVICE
        self.model: Any = None
        self.processor: Any = None
        self.params: str = ""

    @property
    def quantization(self) -> str:
        return "4-bit" if self.use_4bit else "FP16"

    def _load_hf(
        self,
        model_cls=AutoModelForImageTextToText,
        processor_cls=AutoProcessor,
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        token = get_hf_token()
        proc_kw = {"token": token, "trust_remote_code": True}
        if processor_kwargs:
            proc_kw.update(processor_kwargs)
        self.processor = processor_cls.from_pretrained(self.hf_id, **proc_kw)

        mod_kw = {"token": token, "trust_remote_code": True, "device_map": DEVICE}
        if self.use_4bit:
            mod_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            mod_kw["dtype"] = torch.float16

        if model_kwargs:
            mod_kw.update(model_kwargs)

        self.model = model_cls.from_pretrained(self.hf_id, **mod_kw).eval()
        if not self.use_4bit:
            self.model = self.model.to(torch.float16)

    def _run_generate(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int = 150,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        inputs = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prompt_len = inputs["input_ids"].shape[1]
        raw_output = self.processor.decode(
            generated_ids[0][prompt_len:], skip_special_tokens=True
        )
        is_proper, feedback, parse_success = extract_prediction(raw_output)
        return {
            "raw_text": raw_output,
            "is_proper": is_proper,
            "feedback": feedback,
            "parse_success": parse_success,
            "latency_ms": latency_ms,
        }

    @abstractmethod
    def load(self):
        """Load model weights and processor."""

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Run inference on a single image and return structured prediction dict."""

    def unload(self):
        """Unload model and free GPU VRAM."""
        if self.model is not None:
            self.params = format_param_count(count_model_parameters(self.model))
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class StandardVLMAdapter(VLMAdapter):
    """Universal adapter for standard Hugging Face AutoModelForImageTextToText models."""

    def __init__(
        self,
        name: str,
        hf_id: str,
        display_name: str,
        use_4bit: bool,
        style: str,  # "tokenized" (Qwen3, Gemma4) or "two_stage" (LFM, Mistral, MiniCPM, SmolVLM)
        image_size: tuple[int, int] | None,
        system_as_list: bool,
        repetition_penalty: float | None,
        release_date: str,
    ):
        super().__init__(name, hf_id, display_name, use_4bit, release_date=release_date)
        self.style = style
        self.image_size = image_size
        self.system_as_list = system_as_list
        self.repetition_penalty = repetition_penalty

    def load(self):
        print(f"Loading {self.display_name} ({self.hf_id}) [4-bit={self.use_4bit}]...")
        proc_kw = {}
        if "mistral" in self.hf_id.lower():
            proc_kw["fix_mistral_regex"] = True
        self._load_hf(processor_kwargs=proc_kw)

    def generate(
        self,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        rgb_img = image.convert("RGB")
        if self.image_size:
            rgb_img = rgb_img.copy()
            rgb_img.thumbnail(self.image_size)

        sys_content = (
            [{"type": "text", "text": system_prompt}]
            if self.system_as_list
            else system_prompt
        )

        # Some processors encode images directly inside apply_chat_template(tokenize=True),
        # while others only format the prompt string first, then require calling processor(text=..., images=...).
        if self.style == "tokenized":
            messages = [
                {"role": "system", "content": sys_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb_img},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        else:  # "two_stage"
            messages = [
                {"role": "system", "content": sys_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            inputs = self.processor(text=prompt, images=[rgb_img], return_tensors="pt")

        return self._run_generate(
            inputs,
            repetition_penalty=self.repetition_penalty,
        )


class Qwen2_5VLAdapter(VLMAdapter):
    """Adapter for Qwen2.5-VL using qwen_vl_utils.process_vision_info."""

    def __init__(
        self,
        name: str,
        hf_id: str,
        display_name: str,
        use_4bit: bool,
        release_date: str,
    ):
        super().__init__(name, hf_id, display_name, use_4bit, release_date=release_date)

    def load(self):
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info
        print(f"Loading {self.display_name} ({self.hf_id}) [4-bit={self.use_4bit}]...")
        self._load_hf(
            processor_kwargs={"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
        )

    def generate(
        self,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        rgb_img = image.convert("RGB")
        rgb_img.thumbnail((640, 640))
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_img},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return self._run_generate(inputs)


class InternVLAdapter(VLMAdapter):
    """Adapter for OpenGVLab InternVL 3.5 using torchvision pipeline and model.chat()."""

    def __init__(
        self,
        name: str,
        hf_id: str,
        display_name: str,
        release_date: str,
    ):
        super().__init__(
            name, hf_id, display_name, use_4bit=False, release_date=release_date
        )
        self.transform: Any = None

    def load(self):
        # Patch all_tied_weights_keys for compatibility with transformers 5.x
        original_getattr = nn.Module.__getattr__

        def custom_getattr(self_module, name):
            return (
                {}
                if name == "all_tied_weights_keys"
                else original_getattr(self_module, name)
            )

        nn.Module.__getattr__ = custom_getattr

        print(f"Loading {self.display_name} ({self.hf_id})...")
        token = get_hf_token()
        self.processor = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True, token=token
        )
        self.model = (
            AutoModel.from_pretrained(
                self.hf_id,
                dtype=torch.float16,
                trust_remote_code=True,
                token=token,
            )
            .to(DEVICE)
            .eval()
        )

        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def generate(
        self,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        pixel_values = (
            self.transform(image).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        )

        question = f"<image>\n{system_prompt}\n\nTask:\n{user_prompt}"

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            raw_output = self.model.chat(
                self.processor,
                pixel_values,
                question,
                {"max_new_tokens": 128, "do_sample": False},
            )

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_proper, feedback, parse_success = extract_prediction(raw_output)
        return {
            "raw_text": raw_output,
            "is_proper": is_proper,
            "feedback": feedback,
            "parse_success": parse_success,
            "latency_ms": latency_ms,
        }
