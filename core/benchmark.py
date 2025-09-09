import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader

from .base import BaseRunner
from .validator import Validator

class Benchmark(BaseRunner):
    """Benchmarking utilities for YOLO models."""
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.results = {}
        self.validator = Validator(model, **kwargs)

    def benchmark_accuracy(self, dataloader: DataLoader):
        """
        Benchmark model accuracy (mAP) on a test dataset.
        """
        self.results['accuracy'] = self.validator.validate(dataloader)
        return self.results['accuracy']

    @torch.no_grad()
    def benchmark_speed(
        self,
        img_size: int = 640,
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup: int = 10
    ):
        """
        Benchmark model inference speed.
        
        Args:
            img_size: Input image size
            batch_size: Batch size for inference
            num_iterations: Number of iterations to average
            warmup: Number of warmup iterations
        
        Returns:
            Dict with timing statistics
        """
        self.model.eval()

        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)

        self.logger.info("benchmark/warmup", f"Warming up for {warmup} iterations")
        for _ in range(warmup):
            _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        self.logger.info("benchmark/speed", f"Running {num_iterations} iterations")

        for i in range(num_iterations):
            start = time.perf_counter()

            _ = self.model(dummy_input)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)
        results = {
            'mean_ms': np.mean(times), 'std_ms': np.std(times), 'min_ms': np.min(times), 'max_ms':
                np.max(times), 'fps': 1000 / np.mean(times) * batch_size, 'batch_size': batch_size,
            'img_size': img_size, 'device': str(self.device)
        }

        self.results['speed'] = results
        return results

    @torch.no_grad()
    def benchmark_memory(self, img_size: int = 640, batch_size: int = 1):
        """
        Benchmark model memory usage.
        
        Args:
            img_size: Input image size
            batch_size: Batch size
        
        Returns:
            Dict with memory statistics
        """
        if self.device.type != 'cuda':
            return {'error': 'Memory benchmarking only available for CUDA devices'}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)

        mem_before = torch.cuda.memory_allocated()

        self.model.eval()
        _ = self.model(dummy_input)

        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        results = {
            'model_params':
                sum(p.numel() for p in self.model.parameters()),
            'model_size_mb':
                sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024,
            'inference_memory_mb': (mem_after - mem_before) / 1024 / 1024, 'peak_memory_mb':
                peak_mem / 1024 / 1024, 'batch_size':
                    batch_size, 'img_size':
                        img_size
        }

        self.results['memory'] = results
        return results

    def print_results(self):
        """Print benchmark results."""
        if 'speed' in self.results:
            self.logger.info("benchmark/results", "Speed Benchmark Results:")
            self.logger.info("benchmark/separator", "-" * 50)
            for key, value in self.results['speed'].items():
                if isinstance(value, float):
                    self.logger.info(f"benchmark/speed/{key}", f"{value:.2f}")
                else:
                    self.logger.info(f"benchmark/speed/{key}", value)

        if 'memory' in self.results:
            self.logger.info("benchmark/results", "Memory Benchmark Results:")
            self.logger.info("benchmark/separator", "-" * 50)
            for key, value in self.results['memory'].items():
                if isinstance(value, float):
                    self.logger.info(f"benchmark/memory/{key}", f"{value:.2f}")
                else:
                    self.logger.info(f"benchmark/memory/{key}", value)

        if 'accuracy' in self.results:
            self.logger.info("benchmark/results", "Accuracy Benchmark Results:")
            self.logger.info("benchmark/separator", "-" * 50)
            acc_results = self.results['accuracy']
            self.logger.info("benchmark/accuracy/map50_95", f"{acc_results.map50_95:.4f}")
            self.logger.info("benchmark/accuracy/map50", f"{acc_results.map50:.4f}")
            self.logger.info("benchmark/accuracy/map75", f"{acc_results.map75:.4f}")

        self.logger.info("benchmark/separator", "-" * 50)
