"""Load a frozen LLaVA-Med with MedITok forced in as its vision encoder, and ask
it a question about an image.

Wraps the MedITok paper's own downstream integration (arXiv 2505.19225 Sec 4.5):
the released `massaki75/meditok` HF repo ships `llavamed_meditok/` (the VLM) and
`meditok/meditok_simple_v1.pth` (the vision-tower weights). CUDA + fp16 only.

Needs the meditok "understanding" fork on disk
(`<meditok_repo>/evaluation/understanding`, cloned by scripts/setup_third_party.sh)
and its own pinned dependency stack (pyproject `[vqa]` extra). `load_llavamed`
adds that fork to `sys.path` and chdirs into it, because the fork's own
`sample_vqa.sh` resolves several config paths relative to that directory. It
does not restore the cwd.
"""
import json
import os
import sys

import torch
from PIL import Image

from src.generative.vqa_scoring import CLINICAL_QUESTION, MODALITY_Q, REGION_Q

# Questions asked per image. Kermany/APTOS: modality, body region, clinical.
# ISIC has no body-region question; its extra lesion-type question is asked but
# not scored (open-ended, no fixed vocabulary).
QUESTIONS = {
    "kermany": [MODALITY_Q, REGION_Q, CLINICAL_QUESTION["kermany"]],
    "aptos": [MODALITY_Q, REGION_Q, CLINICAL_QUESTION["aptos"]],
    "isic": [MODALITY_Q, CLINICAL_QUESTION["isic"],
             "What type of skin lesion is shown in this image?"],
}


class LlavaMed:
    """Frozen LLaVA-Med(MedITok). `ask(img, question)` greedy-decodes an answer;
    `img` is [3,H,W] or [B,3,H,W], the processor output of `load_image`."""

    def __init__(self, tokenizer, model, image_processor, device: str):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.device = device

    @property
    def vision_tower_core(self):
        """The MedITok encoder inside the VLM, the target of the VQA attack."""
        return self.model.get_vision_tower().vision_tower

    def load_image(self, path: str) -> torch.Tensor:
        from llava.mm_utils import process_images

        return process_images([Image.open(path).convert("RGB")], self.image_processor, self.model.config)[0]

    @torch.inference_mode()
    def ask(self, img: torch.Tensor, question: str, conv_mode: str = "v1", max_new_tokens: int = 64) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to(self.device)
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        crit = KeywordsStoppingCriteria([stop], self.tokenizer, ids)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        out = self.model.generate(
            ids, images=img.half().to(self.device),
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            use_cache=True, stopping_criteria=[crit],
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def load_llavamed(meditok_repo: str, weights_dir: str | None = None, device: str = "cuda") -> LlavaMed:
    understanding_dir = os.path.join(meditok_repo, "evaluation", "understanding")
    weights_dir = weights_dir or os.path.join(meditok_repo, "weights")
    vqa_dir = os.path.join(weights_dir, "llavamed_meditok")
    vt_path = os.path.join(weights_dir, "meditok", "meditok_simple_v1.pth")
    assert os.path.isdir(vqa_dir), f"missing {vqa_dir}; run setup_third_party.sh with MEDITOK_VLM=1"
    assert os.path.isfile(vt_path), f"missing {vt_path}"

    if understanding_dir not in sys.path:
        sys.path.insert(0, understanding_dir)
    os.chdir(understanding_dir)

    cfg_path = os.path.join(vqa_dir, "config.json")
    cfg = json.load(open(cfg_path))
    if cfg.get("mm_vision_tower") != vt_path:
        cfg["mm_vision_tower"], cfg["custom_encoder"], cfg["quantize"] = vt_path, True, True
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=vqa_dir, model_base=None, model_name="llava_mistral_meditok",
        force_vision_tower_path=vt_path, quantize=True, custom_encoder=True,
    )
    return LlavaMed(tokenizer, model, image_processor, device)


def swap_vision_tower(llavamed: LlavaMed, ckpt_path: str) -> None:
    """Beyond-paper: replace the VLM's vision tower with a defended (fine-tuned)
    MedITok checkpoint. Only the tower is swapped. The LLM and mm_projector were
    trained against the original encoder's clean feature distribution, so re-run
    a clean-image sanity check after calling this."""
    from argparse import Namespace

    from llava.model.multimodal_encoder.builder import build_vision_tower

    tower = build_vision_tower(Namespace(mm_vision_tower=ckpt_path, quantize=True, custom_encoder=True))
    tower.to(device=llavamed.device, dtype=torch.float16)
    llavamed.model.model.vision_tower = tower
