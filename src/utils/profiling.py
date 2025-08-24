"""Model profiling and benchmarking utilities."""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from contextlib import contextmanager
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MemoryTracker:
    """GPU and CPU memory tracking."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.snapshots = []
        
    def snapshot(self, tag: str = ""):
        """Take memory snapshot."""
        snapshot = {
            'tag': tag,
            'timestamp': time.time(),
        }
        
        # GPU memory
        if self.device.type == 'cuda':
            snapshot.update({
                'gpu_allocated': torch.cuda.memory_allocated(self.device),
                'gpu_cached': torch.cuda.memory_reserved(self.device),
                'gpu_max_allocated': torch.cuda.max_memory_allocated(self.device),
            })
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        snapshot.update({
            'cpu_rss': memory_info.rss,
            'cpu_vms': memory_info.vms,
        })
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.snapshots:
            return {}
        
        peak = {'cpu_rss': 0, 'cpu_vms': 0}
        if self.device.type == 'cuda':
            peak.update({'gpu_allocated': 0, 'gpu_cached': 0})
        
        for snapshot in self.snapshots:
            for key in peak:
                if key in snapshot:
                    peak[key] = max(peak[key], snapshot[key])
        
        # Convert to MB
        return {k: v / (1024 ** 2) for k, v in peak.items()}
    
    def memory_diff(self, start_tag: str, end_tag: str) -> Dict[str, float]:
        """Compute memory difference between snapshots."""
        start_snapshot = None
        end_snapshot = None
        
        for snapshot in self.snapshots:
            if snapshot['tag'] == start_tag:
                start_snapshot = snapshot
            elif snapshot['tag'] == end_tag:
                end_snapshot = snapshot
        
        if start_snapshot is None or end_snapshot is None:
            return {}
        
        diff = {}
        for key in ['cpu_rss', 'cpu_vms']:
            if key in start_snapshot and key in end_snapshot:
                diff[key] = (end_snapshot[key] - start_snapshot[key]) / (1024 ** 2)
        
        if self.device.type == 'cuda':
            for key in ['gpu_allocated', 'gpu_cached']:
                if key in start_snapshot and key in end_snapshot:
                    diff[key] = (end_snapshot[key] - start_snapshot[key]) / (1024 ** 2)
        
        return diff


class ModelProfiler:
    """Comprehensive model profiling."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.memory_tracker = MemoryTracker(device)
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Count by module type
        param_by_type = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_type = type(module).__name__
                module_params = sum(p.numel() for p in module.parameters())
                param_by_type[module_type] = param_by_type.get(module_type, 0) + module_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params,
            'by_type': param_by_type,
        }
    
    def measure_inference_time(
        self,
        input_batch: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Measure inference time statistics."""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(**input_batch)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = self.model(**input_batch)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
        }
    
    def measure_throughput(
        self,
        input_batch: Dict[str, torch.Tensor],
        duration_seconds: float = 10.0,
    ) -> Dict[str, float]:
        """Measure inference throughput."""
        self.model.eval()
        batch_size = next(iter(input_batch.values())).size(0)
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        num_batches = 0
        
        with torch.no_grad():
            while time.perf_counter() < end_time:
                _ = self.model(**input_batch)
                num_batches += 1
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        actual_duration = time.perf_counter() - start_time
        total_samples = num_batches * batch_size
        
        return {
            'samples_per_second': total_samples / actual_duration,
            'batches_per_second': num_batches / actual_duration,
            'total_samples': total_samples,
            'duration_seconds': actual_duration,
        }
    
    def profile_forward_pass(
        self,
        input_batch: Dict[str, torch.Tensor],
        trace_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Profile forward pass with PyTorch profiler."""
        self.model.eval()
        
        activities = [ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            with record_function("model_inference"):
                _ = self.model(**input_batch)
        
        if trace_file:
            prof.export_chrome_trace(trace_file)
        
        # Extract key statistics
        events = prof.key_averages(group_by_stack_n=5)
        
        cpu_time = sum(event.cpu_time_total for event in events)
        cuda_time = sum(event.cuda_time_total for event in events) if self.device.type == 'cuda' else 0
        
        # Top CPU operations
        cpu_events = sorted(events, key=lambda x: x.cpu_time_total, reverse=True)[:10]
        top_cpu_ops = [(event.key, event.cpu_time_total / 1000) for event in cpu_events]
        
        # Top CUDA operations
        top_cuda_ops = []
        if self.device.type == 'cuda':
            cuda_events = sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:10]
            top_cuda_ops = [(event.key, event.cuda_time_total / 1000) for event in cuda_events]
        
        return {
            'total_cpu_time_ms': cpu_time / 1000,
            'total_cuda_time_ms': cuda_time / 1000,
            'top_cpu_operations': top_cpu_ops,
            'top_cuda_operations': top_cuda_ops,
            'profiler_output': str(prof.key_averages().table(sort_by="cpu_time_total")),
        }
    
    def memory_profile(
        self,
        input_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Profile memory usage during forward pass."""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.memory_tracker.snapshot("before_forward")
        
        with torch.no_grad():
            _ = self.model(**input_batch)
        
        self.memory_tracker.snapshot("after_forward")
        
        memory_stats = {}
        
        if self.device.type == 'cuda':
            memory_stats.update({
                'gpu_peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
                'gpu_peak_cached_mb': torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
            })
        
        memory_diff = self.memory_tracker.memory_diff("before_forward", "after_forward")
        memory_stats.update(memory_diff)
        
        return memory_stats


def profile_model(
    model: nn.Module,
    input_batch: Dict[str, torch.Tensor],
    device: torch.device,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete model profiling."""
    profiler = ModelProfiler(model, device)
    
    logger.info("Starting model profiling...")
    
    # Parameter count
    param_stats = profiler.count_parameters()
    logger.info(f"Total parameters: {param_stats['total']:,}")
    
    # Inference timing
    timing_stats = profiler.measure_inference_time(input_batch)
    logger.info(f"Mean inference time: {timing_stats['mean_ms']:.2f} ms")
    
    # Throughput
    throughput_stats = profiler.measure_throughput(input_batch)
    logger.info(f"Throughput: {throughput_stats['samples_per_second']:.1f} samples/s")
    
    # Memory usage
    memory_stats = profiler.memory_profile(input_batch)
    if 'gpu_peak_allocated_mb' in memory_stats:
        logger.info(f"Peak GPU memory: {memory_stats['gpu_peak_allocated_mb']:.1f} MB")
    
    # Detailed profiling
    trace_file = None
    if output_dir:
        trace_file = f"{output_dir}/trace.json"
    
    profile_stats = profiler.profile_forward_pass(input_batch, trace_file)
    
    results = {
        'parameters': param_stats,
        'timing': timing_stats,
        'throughput': throughput_stats,
        'memory': memory_stats,
        'profiling': profile_stats,
    }
    
    return results


@contextmanager
def benchmark_context(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        logger.info(f"{name}: {(end - start) * 1000:.2f} ms")


def benchmark_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    num_steps: int = 100
) -> Dict[str, Any]:
    """Benchmark training performance."""
    profiler = ModelProfiler(model, device)
    
    # Benchmark forward pass
    forward_stats = profiler.benchmark_forward(train_loader, num_steps=num_steps)
    
    # Benchmark backward pass
    backward_stats = profiler.benchmark_backward(train_loader, num_steps=num_steps)
    
    # Benchmark memory usage
    memory_stats = profiler.benchmark_memory(train_loader, num_steps=num_steps)
    
    return {
        'forward': forward_stats,
        'backward': backward_stats,
        'memory': memory_stats,
        'summary': {
            'total_time': forward_stats['total_time'] + backward_stats['total_time'],
            'throughput': num_steps / (forward_stats['total_time'] + backward_stats['total_time']),
            'peak_memory': memory_stats['peak_memory']
        }
    }


class ProfilerManager:
    """Simple profiler manager for training."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiles = {}
    
    def start_profile(self, name: str):
        """Start profiling a section."""
        if self.enabled:
            self.profiles[name] = {'start_time': time.time()}
    
    def end_profile(self, name: str):
        """End profiling a section."""
        if self.enabled and name in self.profiles:
            self.profiles[name]['end_time'] = time.time()
            self.profiles[name]['duration'] = (
                self.profiles[name]['end_time'] - self.profiles[name]['start_time']
            )
    
    def get_profile(self, name: str) -> Optional[Dict[str, float]]:
        """Get profile results for a section."""
        return self.profiles.get(name)
    
    def get_all_profiles(self) -> Dict[str, Dict[str, float]]:
        """Get all profile results."""
        return self.profiles.copy()
    
    def reset(self):
        """Reset all profiles."""
        self.profiles.clear()


def compare_model_sizes(models: Dict[str, nn.Module]) -> Dict[str, Dict[str, int]]:
    """Compare parameter counts across models."""
    results = {}
    
    for name, model in models.items():
        profiler = ModelProfiler(model, torch.device('cpu'))
        results[name] = profiler.count_parameters()
    
    return results


def print_profiling_summary(results: Dict[str, Any], model_name: str = "Model"):
    """Print formatted profiling summary."""
    print("=" * 80)
    print(f"PROFILING SUMMARY: {model_name}")
    print("=" * 80)
    
    # Parameters
    params = results.get('parameters', {})
    print(f"\nPARAMETERS:")
    print(f"  Total: {params.get('total', 0):,}")
    print(f"  Trainable: {params.get('trainable', 0):,}")
    print(f"  Non-trainable: {params.get('non_trainable', 0):,}")
    
    # Timing
    timing = results.get('timing', {})
    if timing:
        print(f"\nINFERENCE TIMING:")
        print(f"  Mean: {timing.get('mean_ms', 0):.2f} ms")
        print(f"  Std: {timing.get('std_ms', 0):.2f} ms")
        print(f"  P95: {timing.get('p95_ms', 0):.2f} ms")
        print(f"  P99: {timing.get('p99_ms', 0):.2f} ms")
    
    # Throughput
    throughput = results.get('throughput', {})
    if throughput:
        print(f"\nTHROUGHPUT:")
        print(f"  Samples/sec: {throughput.get('samples_per_second', 0):.1f}")
        print(f"  Batches/sec: {throughput.get('batches_per_second', 0):.1f}")
    
    # Memory
    memory = results.get('memory', {})
    if memory:
        print(f"\nMEMORY USAGE:")
        for key, value in memory.items():
            if 'mb' in key.lower():
                print(f"  {key.replace('_', ' ').title()}: {value:.1f} MB")
    
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    import argparse
    from src.models import HierarchicalMSAmba
    from src.utils.config import Config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='hierarchical', help='Model type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--memory', action='store_true', help='Enable memory profiling')
    parser.add_argument('--trace', action='store_true', help='Generate trace file')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # Create model
    model = HierarchicalMSAmba(config.model).to(device)
    
    # Create dummy input
    input_batch = {
        'text_features': torch.randn(args.batch_size, args.seq_len, 768).to(device),
        'audio_features': torch.randn(args.batch_size, args.seq_len, 74).to(device),
        'vision_features': torch.randn(args.batch_size, args.seq_len, 47).to(device),
        'attention_mask': torch.ones(args.batch_size, args.seq_len).to(device),
    }
    
    # Profile model
    results = profile_model(
        model, 
        input_batch, 
        device,
        output_dir='profiling_output' if args.trace else None
    )
    
    # Print results
    print_profiling_summary(results, args.model)