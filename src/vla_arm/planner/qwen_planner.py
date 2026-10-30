"""Qwen3.5-9B 기반 고수준 Task Planner.

- 입력: 사용자 지시문 + 상단/측면 카메라 이미지
- 출력: VLA 가 실행할 subtask 문장 리스트 (JSON)
- 16GB GPU 에서 VLA(4B, bf16) 와 함께 올리기 위해 기본적으로 4bit(NF4) 로 로드
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

SYSTEM_PROMPT = """You are the task planner of a Franka Panda robot arm on a tabletop.
You see camera images of the scene and receive a user instruction.
Decompose the instruction into a sequence of primitive subtasks for a low-level visuomotor policy.

Allowed subtask templates (use them verbatim, fill in object and target names exactly):
  - "pick up the <object>"
  - "place the <object> on the <target>"
  - "move to the home position"
Possible objects on the table: red cube, green cube, blue cube, yellow cube, purple cylinder.
Possible targets on the table: red plate, yellow plate, blue plate, purple plate.
If asked to sort multiple objects onto matching colored plates, output the pick and place pair for each object in order.

Answer ONLY with JSON: {"subtasks": ["...", "..."]}"""

CHECK_PROMPT = """Current subtask: "{subtask}".
Look at the images. Has the subtask been completed? Answer ONLY with JSON: {{"done": true}} or {{"done": false}}"""


@dataclass
class PlannerConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    image_size: int = 384  # planner 에 주는 이미지 한 변 (토큰 수 조절)
    device_map: str = "cuda:0"


def _to_pil(img: np.ndarray | Image.Image, size: int) -> Image.Image:
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    return im.resize((size, size), Image.BICUBIC) if im.size != (size, size) else im


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"planner output has no JSON: {text!r}")
    return json.loads(m.group(0))


class QwenPlanner:
    def __init__(self, cfg: PlannerConfig | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.cfg = cfg or PlannerConfig()
        quant = None
        if self.cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # 비전 인코더/출력층은 양자화하지 않음 (정확도 보존)
                llm_int8_skip_modules=["visual", "lm_head", "mtp"],
            )
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        # WSL: 9B 로드 시 transformers 의 비동기(멀티스레드) 가중치 로딩이
        # "CUDA driver error: device not ready" 로 실패 -> 동기 로딩 + warmup 비활성화
        import os

        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        import transformers.modeling_utils as _mu

        _mu.caching_allocator_warmup = lambda *a, **k: None
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id,
            dtype=torch.bfloat16,
            quantization_config=quant,
            device_map=self.cfg.device_map,
        ).eval()

    @torch.inference_mode()
    def _generate(self, images: list, text: str, system: str | None = SYSTEM_PROMPT) -> str:
        content = [{"type": "image", "image": _to_pil(im, self.cfg.image_size)} for im in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        if system:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            enable_thinking=False,  # 실시간 제어를 위해 thinking 모드 비활성화
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def plan(self, instruction: str, images: dict) -> list[str]:
        """images: {"top": ndarray, "side": ndarray, ...} — 상단/측면 이미지를 사용."""
        views = [images[k] for k in ("top", "side") if k in images]
        raw = self._generate(views, f"Instruction: {instruction}")
        try:
            subtasks = _extract_json(raw)["subtasks"]
            assert isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks)
        except Exception as e:  # noqa: BLE001
            print(f"[planner] parse failed ({e}); raw={raw!r}")
            subtasks = [instruction]
        return subtasks

    def check_done(self, subtask: str, images: dict) -> bool:
        views = [images[k] for k in ("top", "side", "wrist") if k in images]
        raw = self._generate(views, CHECK_PROMPT.format(subtask=subtask))
        try:
            return bool(_extract_json(raw)["done"])
        except Exception:  # noqa: BLE001
            return False

    def unload(self):
        import gc

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
