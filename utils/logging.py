import os
import json
import torch
import numpy as np
from enum import IntEnum
import threading
import queue
from time import perf_counter
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Any, Optional, Union, Dict, List
try:
    from utils.helpers import suppress_stderr_fd
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def suppress_stderr_fd():
        import sys
        devnull = open(os.devnull, "w")
        old = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old
            devnull.close()

try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    with suppress_stderr_fd():
        from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    BASIC = 50  # For basic metrics like mAP, losses
    HEAVY = 60  # For detailed weight/gradient stats (histograms, images)

class YOLOLogger:
    def __init__(
        self,
        log_dir: Union[str, Path] = "runs/exp",
        level: Union[LogLevel, List[LogLevel], str, int] = LogLevel.INFO,
        console: bool = True,
        tensorboard: bool = True,
        flush_always: bool = True,
        tb_only: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console_min_level: LogLevel = LogLevel.INFO
        self.tb_basic_enabled: bool = True
        self.tb_heavy_enabled: bool = False
        self.apply_level_tokens(level)
        self.tb_only = bool(tb_only)
        self.console = bool(console and (not self.tb_only))
        env_tb = os.getenv("YOLO_TENSORBOARD", "").strip().lower()
        tb_enabled_env = not (env_tb in ("0", "false", "no"))
        self.tensorboard = tensorboard and tb_enabled_env and SummaryWriter is not None
        self.flush_always = flush_always
        self.step = 0
        self.rank = int(os.getenv("RANK", "-1"))
        self.is_main_process = self.rank in (-1, 0)
        self.writer = None
        if self.tensorboard and self.is_main_process:
            try:
                self.writer = SummaryWriter(str(self.log_dir))
            except Exception:
                self.tensorboard = False
                self.writer = None
        self.log_file = self.log_dir / "log.txt"
        self.jsonl_file = self.log_dir / "log.jsonl"
        self._lock = Lock()
        self._t0 = perf_counter()
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=10000)
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._run, name="yolo-logger", daemon=True)
        self._worker.start()
        self._write_text("logger initialized", LogLevel.INFO)

    def _parse_tokens(self, level_in: Union[LogLevel, List[LogLevel], str, int]) -> List[str]:
        tokens: List[str] = []
        try:
            if isinstance(level_in, LogLevel):
                tokens = [level_in.name]
            elif isinstance(level_in, int):
                for name in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    if int(LogLevel[name]) == int(level_in):
                        tokens = [name]
                        break
            elif isinstance(level_in, str):
                txt = level_in.strip()
                tokens = [p.strip().upper() for p in txt.split(',') if p.strip()]
            elif isinstance(level_in, (list, tuple)):
                for l in level_in:
                    if isinstance(l, LogLevel):
                        tokens.append(l.name)
                    elif isinstance(l, str):
                        tokens.append(l.upper())
        except Exception:
            tokens = []
        return tokens

    def apply_level_tokens(self, level_in: Union[LogLevel, List[LogLevel], str, int]):
        """
        Parse level tokens and configure filters.
        - CLI tokens: DEBUG, INFO, WARNING, ERROR -> set threshold (default INFO)
        - TB tokens: BASIC (default enabled), HEAVY (enables heavy + basic)
        """
        console_min = LogLevel.INFO
        tb_basic = True
        tb_heavy = False

        tokens = self._parse_tokens(level_in)
        cli_tokens = [t for t in tokens if t in ("DEBUG", "INFO", "WARNING", "ERROR")]
        if cli_tokens:
            try:
                console_min = min(LogLevel[t] for t in cli_tokens)
            except Exception:
                console_min = LogLevel.INFO
        if "HEAVY" in tokens:
            tb_heavy = True
            tb_basic = True
        elif "BASIC" in tokens:
            tb_basic = True

        self.console_min_level = console_min
        self.tb_basic_enabled = tb_basic
        self.tb_heavy_enabled = tb_heavy

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _should_log_console(self, level: LogLevel) -> bool:
        if level not in (LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR):
            return False
        try:
            return level >= self.console_min_level
        except Exception:
            return True

    def _format_line(self, text: str, level: LogLevel) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"[{ts}] [{level.name:5s}] {text}"

    def _write_text(self, text: str, level: LogLevel):
        if (not self.tb_only) and (not self._should_log_console(level)):
            return
        line = self._format_line(text, level)
        if level == LogLevel.INFO:
            try:
                if self.console and self.is_main_process:
                    print(line, flush=self.flush_always)
            except Exception:
                pass
            if not self.tb_only:
                try:
                    with self._lock:
                        with open(self.log_file, "a", encoding="utf-8") as f:
                            f.write(line + "\n")
                except Exception:
                    pass
                return
            else:
                try:
                    self._q.put_nowait({"type": "text", "line": line, "level": level})
                except queue.Full:
                    pass
                return
        try:
            self._q.put_nowait({"type": "text", "line": line, "level": level})
        except queue.Full:
            pass

    def _log_jsonl(self, record: Dict[str, Any]):
        if not self.is_main_process:
            return
        try:
            self._q.put_nowait({"type": "jsonl", "record": record})
        except queue.Full:
            pass

    def _tb_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            try:
                self._q.put_nowait({
                    "type": "tb_scalar", "tag": tag, "value": float(value), "step": int(step)
                })
            except queue.Full:
                pass

    def _tb_hist(self, tag: str, tensor: Union[torch.Tensor, np.ndarray], step: int):
        if self.writer:
            try:
                self._q.put_nowait({
                    "type": "tb_hist", "tag": tag, "tensor": tensor, "step": int(step)
                })
            except queue.Full:
                pass

    def _tb_text(self, tag: str, text: str, step: int):
        if self.writer:
            try:
                self._q.put_nowait({"type": "tb_text", "tag": tag, "text": text, "step": int(step)})
            except queue.Full:
                pass

    def _tb_image(self, tag: str, tensor: torch.Tensor, step: int):
        if self.writer:
            try:
                self._q.put_nowait({
                    "type": "tb_image", "tag": tag, "tensor": tensor, "step": int(step)
                })
            except queue.Full:
                pass

    def _log_to_tensorboard(self, tag: str, data: Any, step: int):
        if not (self.writer and self.tensorboard and self.is_main_process):
            return
        try:
            constant_prefixes = ("config/", "run/", "dataloader/", "model/")
            if isinstance(data, dict):
                if any(tag.startswith(p) for p in constant_prefixes):
                    import json
                    self._tb_text(tag, json.dumps(data, default=str, indent=2), step)
                    return
            if isinstance(data, (int, float, np.number)):
                self._tb_scalar(tag, float(data), step)
            elif isinstance(data, torch.Tensor):
                if data.numel() == 1:
                    self._tb_scalar(tag, float(data.item()), step)
                elif data.dim() in (1, 2) and ("weight" in tag or "grad" in tag or "param" in tag):
                    self._tb_hist(tag, data, step)
                elif data.dim() == 2:
                    self._tb_hist(tag, data, step)
                elif data.dim() == 3:
                    self._tb_image(tag, data, step)
                elif data.dim() == 4:
                    self._tb_image(tag, data, step)
                else:
                    flat = data.reshape(-1)
                    self._tb_hist(f"{tag}/values", flat, step)
                    self._tb_scalar(f"{tag}/mean", float(flat.float().mean().item()), step)
                    self._tb_scalar(
                        f"{tag}/std", float(flat.float().std(unbiased=False).item()), step
                    )
                    self._tb_scalar(f"{tag}/min", float(flat.min().item()), step)
                    self._tb_scalar(f"{tag}/max", float(flat.max().item()), step)
            elif isinstance(data, np.ndarray):
                self._log_to_tensorboard(tag, torch.from_numpy(data), step)
            elif isinstance(data, dict):
                for k, v in data.items():
                    self._log_to_tensorboard(f"{tag}/{k}", v, step)
            elif isinstance(data, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in data):
                    self._tb_hist(tag, np.asarray(data), step)
                else:
                    for i, v in enumerate(data):
                        self._log_to_tensorboard(f"{tag}/{i}", v, step)
            else:
                self._tb_text(tag, str(data), step)
        except Exception as e:
            self._write_text(f"tensorboard log failed for {tag}: {e}", LogLevel.DEBUG)

    def log_text(self, tag: str, text: str, step: int | None = None):
        s = self.step if step is None else step
        if self.writer and self.tensorboard and self.is_main_process:
            self._tb_text(tag, text, s)

    def log(
        self,
        tag: str,
        data: Any,
        level: LogLevel = LogLevel.INFO,
        step: Optional[int] = None,
        console_msg: Optional[str] = None
    ):
        if step is not None:
            self.step = step
        if level in (LogLevel.BASIC, LogLevel.HEAVY):
            if (level == LogLevel.BASIC and not self.tb_basic_enabled
               ) or (level == LogLevel.HEAVY and not self.tb_heavy_enabled):
                return
            self._log_to_tensorboard(tag, data, self.step)
            return

        if (not self.tb_only) and (not self._should_log_console(level)):
            return
        rec = {
            "time": datetime.now().isoformat(), "elapsed_s": round(perf_counter() - self._t0, 3),
            "step": int(self.step), "level": level.name, "tag": tag
        }
        if isinstance(data, (int, float, str)):
            rec["data"] = data
        elif isinstance(data, torch.Tensor) and data.numel() == 1:
            rec["data"] = float(data.item())
        else:
            rec["data"] = str(data)
        self._log_jsonl(rec)
        msg = console_msg if console_msg is not None else rec["data"]
        self._write_text(f"{tag}: {msg}", level)

    def debug(self, tag: str, data: Any, **kwargs):
        self.log(tag, data, LogLevel.DEBUG, **kwargs)

    def info(self, tag: str, data: Any, **kwargs):
        self.log(tag, data, LogLevel.INFO, **kwargs)

    def basic(self, tag: str, data: Any, **kwargs):
        """Log basic metrics like mAP50-95, cls loss, dfl loss, box loss."""
        self.log(tag, data, LogLevel.BASIC, **kwargs)

    def warning(self, tag: str, data: Any, **kwargs):
        self.log(tag, data, LogLevel.WARNING, **kwargs)

    def error(self, tag: str, data: Any, **kwargs):
        self.log(tag, data, LogLevel.ERROR, **kwargs)

    def heavy(self, tag: str, data: Any, **kwargs):
        self.log(tag, data, LogLevel.HEAVY, **kwargs)

    def set_step(self, step: int):
        self.step = int(step)

    def log_images_with_boxes(
        self,
        tag: str,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        return self.log_images_with_boxes_async(tag, images, boxes=boxes, labels=labels, step=step)

    def log_images_with_boxes_async(
        self,
        tag: str,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[List[str]] = None,
        step: Optional[int] = None,
    ):
        """Enqueue a builder that draws boxes and logs images in the worker thread (BASIC)."""
        if not (
            self.writer and self.tensorboard and self.is_main_process and self.tb_basic_enabled
        ):
            return
        s = self.step if step is None else step
        try:
            if images.dim() == 4:
                imgs = images[:4].detach().cpu()
            else:
                imgs = images.detach().cpu()
            if isinstance(boxes, torch.Tensor):
                bxs = boxes[:4].detach().cpu() if boxes.dim() >= 2 else boxes.detach().cpu()
            else:
                bxs = None

            def _builder():
                try:
                    from torchvision.utils import draw_bounding_boxes
                except Exception:
                    return imgs
                b = imgs
                if b.dtype != torch.uint8:
                    b = (b.clamp(0, 1) * 255).to(torch.uint8)
                if b.dim() == 3:
                    b = b.unsqueeze(0)
                if bxs is None:
                    return b
                out = []
                for i in range(b.shape[0]):
                    bi = b[i]
                    bb = bxs if (isinstance(bxs, torch.Tensor) and bxs.dim() == 2) else (
                        bxs[i] if isinstance(bxs, torch.Tensor) else
                        torch.empty((0, 4), dtype=torch.float32)
                    )
                    try:
                        out.append(
                            draw_bounding_boxes(
                                bi, bb, labels=labels if labels is not None else None
                            )
                        )
                    except Exception:
                        out.append(bi)
                if len(out):
                    return torch.stack(out, 0)
                return b

            try:
                self._q.put_nowait({
                    "type": "tb_image_build", "tag": tag, "builder": _builder, "step": int(s)
                })
            except queue.Full:
                pass
        except Exception as e:
            self._write_text(f"image box async log failed: {e}", LogLevel.DEBUG)

    def log_figure(self, tag: str, fig, step: int | None = None):
        s = self.step if step is None else step
        if self.writer and self.tensorboard and self.is_main_process:
            try:
                self._q.put_nowait({"type": "tb_figure", "tag": tag, "fig": fig, "step": int(s)})
            except queue.Full:
                pass

    def log_figure_async(self, tag: str, builder, step: int | None = None):
        """Enqueue a callable that builds and returns a matplotlib figure (BASIC)."""
        s = self.step if step is None else step
        if self.writer and self.tensorboard and self.is_main_process and self.tb_basic_enabled:
            try:
                self._q.put_nowait({
                    "type": "tb_figure_build", "tag": tag, "builder": builder, "step": int(s)
                })
            except queue.Full:
                pass

    def log_scalars_group(self, tag: str, scalars: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalars under a single group (SummaryWriter.add_scalars). Treated as BASIC."""
        if not (
            self.writer and self.tensorboard and self.is_main_process and self.tb_basic_enabled
        ):
            return
        s = self.step if step is None else step
        try:
            self._q.put_nowait({
                "type": "tb_scalars", "tag": tag, "scalars": scalars, "step": int(s)
            })
        except queue.Full:
            pass

    def log_losses(self, losses: Dict[str, float], step: Optional[int] = None):
        s = self.step if step is None else step
        for k, v in losses.items():
            if k in ['box', 'cls', 'dfl', 'obj']:  # Core losses
                self.basic(f"loss/{k}", float(v), step=s)
            else:
                self.info(f"loss/{k}", float(v), step=s)
        tot = float(sum(losses.values())) if losses else 0.0
        parts = ", ".join(f"{k}:{float(v):.4f}" for k, v in losses.items())
        self.basic("loss/total", tot, step=s)
        self.info("loss/summary", f"{tot:.4f} ({parts})", step=s)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        s = self.step if step is None else step
        for k, v in metrics.items():
            if 'mAP' in k or 'fitness' in k:
                self.basic(f"metrics/{k}", float(v), step=s)
            else:
                self.info(f"metrics/{k}", float(v), step=s)

        parts = ", ".join(f"{k}:{float(v):.4f}" for k, v in metrics.items())
        self.info("metrics/summary", parts, step=s, console_msg=parts)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        s = self.step if step is None else step
        for i, g in enumerate(optimizer.param_groups):
            self.debug(f"lr/group_{i}", float(g.get("lr", 0.0)), step=s)

    def close(self):
        try:
            self._stop_evt.set()
            self._q.put_nowait({"type": "_stop"})
        except Exception:
            pass
        try:
            if hasattr(self, "_worker") and self._worker.is_alive():
                self._worker.join(timeout=2.0)
        except Exception:
            pass
        if self.writer:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
        try:
            line = self._format_line("logger closed", LogLevel.INFO)
            if self.console and self.is_main_process:
                print(line, flush=True)
            if not self.tb_only:
                with self._lock:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
        except Exception:
            pass

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                rec = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("type") == "_stop":
                break
            try:
                if rec.get("type") == "text":
                    line = rec.get("line", "")
                    level = rec.get("level", LogLevel.INFO)
                    if self.console and self.is_main_process and self._should_log_console(level):
                        print(line, flush=self.flush_always)
                    if not self.tb_only:
                        with self._lock:
                            with open(self.log_file, "a", encoding="utf-8") as f:
                                f.write(line + "\n")
                    else:
                        if self.writer and self.is_main_process:
                            try:
                                self.writer.add_text(
                                    f"console/{getattr(level, 'name', 'INFO')}", line,
                                    int(self.step)
                                )
                                if self.flush_always:
                                    self.writer.flush()
                            except Exception:
                                pass
                elif rec.get("type") == "jsonl" and self.is_main_process and (not self.tb_only):
                    with self._lock:
                        with open(self.jsonl_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec.get("record", {}), default=str) + "\n")
                elif rec.get("type") == "tb_scalar" and self.writer:
                    self.writer.add_scalar(rec["tag"], rec["value"], rec["step"])
                    if self.flush_always:
                        self.writer.flush()
                elif rec.get("type") == "tb_scalars" and self.writer:
                    try:
                        self.writer.add_scalars(rec["tag"], rec["scalars"], rec["step"])
                        if self.flush_always:
                            self.writer.flush()
                    except Exception as e:
                        self._write_text(f"scalars group log failed: {e}", LogLevel.DEBUG)
                elif rec.get("type") == "tb_hist" and self.writer:
                    self.writer.add_histogram(rec["tag"], rec["tensor"], rec["step"])
                    if self.flush_always:
                        self.writer.flush()
                elif rec.get("type") == "tb_text" and self.writer:
                    self.writer.add_text(rec["tag"], rec["text"], rec["step"])
                    if self.flush_always:
                        self.writer.flush()
                elif rec.get("type") == "tb_image" and self.writer:
                    tensor = rec["tensor"]
                    if tensor.dim() == 3:
                        self.writer.add_image(rec["tag"], tensor, rec["step"])
                    elif tensor.dim() == 4:
                        self.writer.add_images(rec["tag"], tensor, rec["step"])
                    if self.flush_always:
                        self.writer.flush()
                elif rec.get("type") == "tb_figure" and self.writer:
                    self.writer.add_figure(rec["tag"], rec["fig"], rec["step"], close=True)
                    if self.flush_always:
                        self.writer.flush()
                elif rec.get("type") == "tb_figure_build" and self.writer:
                    try:
                        fig = rec["builder"]()
                        self.writer.add_figure(rec["tag"], fig, rec["step"], close=True)
                        if self.flush_always:
                            self.writer.flush()
                    except Exception as e:
                        self._write_text(f"figure build failed: {e}", LogLevel.DEBUG)
                elif rec.get("type") == "tb_image_build" and self.writer:
                    try:
                        tensor = rec["builder"]()
                        if isinstance(tensor, torch.Tensor):
                            if tensor.dim() == 3:
                                self.writer.add_image(rec["tag"], tensor, rec["step"])
                            elif tensor.dim() == 4:
                                self.writer.add_images(rec["tag"], tensor, rec["step"])
                            if self.flush_always:
                                self.writer.flush()
                    except Exception as e:
                        self._write_text(f"image build failed: {e}", LogLevel.DEBUG)
            except Exception:
                pass

_logger: Optional[YOLOLogger] = None

def get_logger(log_dir: Optional[str] = None, **kwargs) -> YOLOLogger:
    global _logger
    if _logger is None:
        _logger = YOLOLogger(log_dir or "runs/exp", **kwargs)
    return _logger

def set_log_level(level: Union[LogLevel, str, int, List[Union[LogLevel, str]]]):
    global _logger
    if _logger is None:
        return
    _logger.apply_level_tokens(level)
