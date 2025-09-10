from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from utils.logging import get_logger

try:
    from torch._dynamo import disable as dynamo_disable  # type: ignore
except Exception:  # pragma: no cover - Dynamo may be unavailable

    def dynamo_disable(fn):
        return fn

class DINOTeacher:
    """
    Lightweight wrapper around HuggingFace DINOv3 ViT models.

    - Lazily imports transformers/torchao if available.
    - Optionally applies weight-only int4 quantization (torchao) to reduce memory/bandwidth.
    - Provides patch tokens, CLS embedding, and a saliency map derived from CLS attention if available
      (falls back to token magnitude if attentions are not returned).
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        resolution: int = 448,
        quant: str = "int4",  # "int8" | "int4" | "fp16" | "fp32" | "bf16"
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
        sal_from: str = "auto",  # "attn" | "energy" | "auto"
        compile: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.resolution = int(resolution)
        self.quant = (quant or "int4").lower().strip()
        self.dtype = dtype

        self._logger = get_logger()
        self._model = None
        self._processor = None
        self._init_ok = False
        self._warned = False
        self._token = hf_token
        self._last_error: Optional[str] = None
        self.sal_from = (sal_from or "auto").lower().strip()
        if self.sal_from not in ("auto", "attn", "energy"):
            self.sal_from = "auto"
        self._auto_peak_mult: float = 5.0  # if max prob <= (k / N) treat as flat
        self._auto_entropy_thresh: float = 0.98  # if H/ln(N) >= thresh treat as flat
        # Runtime flags
        self._quant_applied: bool = False
        self._compiled: bool = False
        self._compile_requested: bool = bool(compile)

    def _lazy_init(self):
        if self._init_ok:
            return
        try:
            from transformers import AutoModel, AutoImageProcessor
        except Exception as e:
            if not self._warned:
                self._logger.warning("dino/missing_transformers", f"{e}")
                self._warned = True
            return

        quantization_config = None
        _manual_weight_only = False
        if self.quant in ("int4", "int8"):
            if self.device.type != "cuda":
                msg = (
                    f"TorchAO {self.quant} quantization requires a CUDA device. "
                    f"Current device is '{self.device}'. Set device='cuda' or choose fp16/fp32/bf16."
                )
                self._last_error = f"{self.quant}_requires_cuda"
                raise RuntimeError(msg)
            try:
                if self.quant == "int4":
                    from torchao.quantization import Int4WeightOnlyConfig as _Cfg  # type: ignore
                    quantization_config = _Cfg(group_size=128)
                else:
                    try:
                        from torchao.quantization import Int8WeightOnlyConfig as _Cfg  # type: ignore
                        quantization_config = _Cfg(group_size=128)
                    except Exception as e:
                        self._logger.warning("dino/int8_unavailable", f"{e}")
                        quantization_config = None
                _manual_weight_only = quantization_config is not None
            except Exception as e:
                self._last_error = f"torchao_config_error: {e}"
                raise

        try:
            if self._token:
                try:
                    self._processor = AutoImageProcessor.from_pretrained(
                        self.model_name, token=self._token
                    )
                except TypeError:
                    self._processor = AutoImageProcessor.from_pretrained(
                        self.model_name, use_auth_token=self._token
                    )
            else:
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception as e:
            self._logger.warning("dino/processor_load_error", str(e))
            self._processor = None
            self._last_error = f"processor_load_error: {e}"

        def _load_model(qcfg):
            kwargs: Dict[str, Any] = {
                "torch_dtype": self.dtype,
                "trust_remote_code": True,
                "device_map": None,
            }
            if self._token:
                try:
                    return AutoModel.from_pretrained(self.model_name, token=self._token, **kwargs)
                except TypeError:
                    return AutoModel.from_pretrained(self.model_name, use_auth_token=self._token, **kwargs)
            else:
                return AutoModel.from_pretrained(self.model_name, **kwargs)

        try:
            if self.quant == "fp16":
                self.dtype = torch.float16
            elif self.quant == "fp32":
                self.dtype = torch.float32
            elif self.quant in ("int4", "int8"):
                self.dtype = self.dtype if self.dtype is not None else torch.bfloat16

            self._model = _load_model(None)
            self._model.to(self.device)
            self._model.eval().requires_grad_(False)
            if _manual_weight_only:
                try:
                    from torchao.quantization import quantize_ as _quantize_
                    _quantize_(self._model, quantization_config)
                    self._quant_applied = True
                except Exception as e:
                    self._last_error = f"torchao_weight_only_apply_error: {e}"
                    raise
            # Optional torch.compile for inference speed (safe defaults)
            if self._compile_requested:
                try:
                    self._model = torch.compile(
                        self._model,
                        backend="inductor",
                        fullgraph=False,
                        dynamic=False,
                    )
                    self._compiled = True
                except Exception as e:
                    # Non-fatal; just report and continue uncompiled
                    try:
                        self._logger.warning("dino/compile_failed", str(e))
                    except Exception:
                        pass
            self._init_ok = True
            self._logger.info(
                "dino/loaded", {
                    "name": self.model_name,
                    "quant": self.quant,
                    "dtype": str(self.dtype),
                    "quant_applied": bool(self._quant_applied),
                    "compiled": bool(self._compiled),
                }
            )
        except Exception as e:
            self._logger.warning("dino/model_load_error", str(e))
            self._model = None
            self._last_error = f"model_load_error: {e}"
            raise

    def unload(self):
        """Release teacher model/processor and free GPU memory if any."""
        try:
            if self._model is not None:
                try:
                    self._model.to("cpu")
                except Exception:
                    pass
            self._model = None
            self._processor = None
            self._init_ok = False
            try:
                if self.device and str(self.device).startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                self._logger.info("dino/unloaded", "Teacher model released")
            except Exception:
                pass
        except Exception as e:
            try:
                self._logger.debug("dino/unload_error", str(e))
            except Exception:
                pass

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Run teacher on images and return tokens/cls/saliency.

        Args:
            images: [B,3,H,W] in [0,1], on the current runtime device.
        Returns:
            dict with keys:
              - tokens: [B, Ht, Wt, Ct]
              - cls: [B, Ct]
              - saliency: [B, Ht, Wt] (probability-like, unnormalized)
            or None if teacher not available
        """
        if self._model is None:
            self._lazy_init()
        if self._model is None:
            return None

        B, C, H, W = images.shape
        pixel_values = self._simple_preprocess(
            images.to(self.device, non_blocking=True), self.resolution
        ).to(device=self.device, dtype=self.dtype)

        with torch.autocast(
            device_type=("cuda" if self.device.type.startswith("cuda") else "cpu"),
            enabled=self.dtype in (torch.float16, torch.bfloat16)
        ):
            out = self._model(pixel_values, output_attentions=True)

        hidden = getattr(out, "last_hidden_state", None)
        pool = getattr(out, "pooler_output", None)
        atts = getattr(out, "attentions", None)
        if hidden is None:
            return None

        cfg = getattr(self._model, "config", None)
        patch_size = int(getattr(cfg, "patch_size", 16))
        num_registers_cfg = int(getattr(cfg, "num_register_tokens", 0))

        try:
            Hin, Win = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        except Exception:
            Hin = Win = int(self.resolution)
        Ht = max(1, Hin // patch_size)
        Wt = max(1, Win // patch_size)
        num_patches = Ht * Wt

        L = hidden.shape[1]
        Ct = hidden.shape[-1]

        has_cls = False
        num_registers = num_registers_cfg

        if L == num_patches + 1 + num_registers:
            has_cls = True
        elif L == num_patches + 1:  # CLS only
            has_cls = True
            num_registers = 0
        elif L == num_patches + num_registers:
            has_cls = False
        elif L == num_patches:
            has_cls = False
            num_registers = 0
        else:
            has_cls = True if L > num_patches else False
            extra = max(0, L - (num_patches + (1 if has_cls else 0)))
            num_registers = min(extra, num_registers if num_registers_cfg else 4)

        if pool is not None:
            cls_vec = pool
        elif has_cls and L >= 1:
            cls_vec = hidden[:, 0]
        else:
            cls_vec = hidden.mean(dim=1)

        seq = hidden
        if has_cls:
            seq = seq[:, 1:, :]

        if num_registers > 0 and seq.shape[1] >= num_registers + num_patches:
            seq = seq[:, num_registers:num_registers + num_patches, :]
        else:
            if seq.shape[1] >= num_patches:
                seq = seq[:, -num_patches:, :]

        if seq.shape[1] != num_patches:
            seq = seq[:, :num_patches, :]

        tokens = seq.reshape(B, Ht, Wt, Ct).contiguous()

        sal_attn = None
        if isinstance(atts, (list, tuple)) and len(atts) > 0:
            try:
                A = atts[-1]  # [B, h, L_att, L_att]
                if A.dim() == 4:
                    A_mean = A.mean(dim=1)  # [B, L_att, L_att]
                    if has_cls and A_mean.shape[-1] >= 1:
                        v = A_mean[:, 0]  # [B, L_att]
                        v = v[:, 1:]  # drop self-CLS
                        if num_registers > 0 and v.shape[1] >= num_registers + num_patches:
                            v = v[:, num_registers:num_registers + num_patches]
                        else:
                            v = v[:, -num_patches:]
                        sal_attn = v.reshape(B, Ht, Wt)
            except Exception:
                sal_attn = None

        sal_energy = tokens.float().pow(2).sum(dim=-1).sqrt()

        mode = self.sal_from
        sal_out = None
        sal_src = ""
        if mode == "energy":
            sal_out = sal_energy
            sal_src = "energy"
        elif mode == "attn":
            sal_out = sal_attn if sal_attn is not None else sal_energy
            sal_src = "attn" if sal_attn is not None else "energy"
        else:  # auto
            if sal_attn is None:
                sal_out = sal_energy
                sal_src = "auto/energy"
            else:
                N = int(Ht * Wt)
                eps = 1e-12
                vtok = sal_attn.view(B, N)
                psum = vtok.sum(dim=1, keepdim=True).clamp_min(eps)
                p = vtok / psum
                pmax = p.max(dim=1).values  # [B]
                ent = -(p * (p + eps).log()).sum(dim=1)  # [B]
                lnN = torch.log(torch.tensor(float(max(2, N)), device=p.device))
                ent_norm = ent / lnN
                thresh = float(self._auto_peak_mult) / float(max(1, N))
                flat_mask = (pmax <= thresh) | (ent_norm >= self._auto_entropy_thresh)
                if flat_mask.any():
                    sal_out = sal_attn.clone()
                    sal_out[flat_mask] = sal_energy[flat_mask]
                    sal_src = "auto/mixed" if (~flat_mask).any() else "auto/energy"
                else:
                    sal_out = sal_attn
                    sal_src = "auto/attn"

        try:
            if not self._warned:
                self._logger.debug("dino/sal_source", sal_src)
        except Exception:
            pass

        return {"tokens": tokens, "cls": cls_vec, "saliency": sal_out, "sal_src": sal_src}

    def _simple_preprocess(self, images: torch.Tensor, size: int) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device,
                            dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device,
                           dtype=images.dtype).view(1, 3, 1, 1)
        imgs = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        imgs = (imgs - mean) / std
        return imgs
