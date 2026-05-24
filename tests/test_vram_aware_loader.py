"""
tests/test_vram_aware_loader.py
--------------------------------
Unit tests for visione.services.common.vram_aware_loader

Run with:
    python -m pytest tests/test_vram_aware_loader.py -v
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Make the common package importable without installing the whole project
# ---------------------------------------------------------------------------
SERVICES_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'visione', 'services'
)
sys.path.insert(0, os.path.abspath(SERVICES_ROOT))

# Provide a minimal stub for visione.extractor so the loader can be imported
# even without the full package installed.
visione_stub = types.ModuleType('visione')
visione_stub.extractor = types.ModuleType('visione.extractor')
sys.modules.setdefault('visione', visione_stub)
sys.modules.setdefault('visione.extractor', visione_stub.extractor)

from common.vram_aware_loader import (  # noqa: E402
    probe_vram,
    suggest_batch_size,
    get_offload_dir,
    load_model_with_offload,
    load_hf_model_with_offload,
    load_hub_model_with_offload,
    load_checkpoint_to_device,
)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model():
    """Return a tiny linear model for test purposes."""
    model = nn.Linear(4, 2)
    return model


# ---------------------------------------------------------------------------
# Tests: probe_vram
# ---------------------------------------------------------------------------

class TestProbeVram(unittest.TestCase):

    def setUp(self):
        # Clean slate for every test
        for key in ('VISIONE_VRAM_THRESHOLD', 'VISIONE_OFFLOAD_MODE'):
            os.environ.pop(key, None)

    def test_returns_cpu_when_cuda_unavailable(self):
        with patch('torch.cuda.is_available', return_value=False):
            result = probe_vram(1.0)
        self.assertEqual(result, 'cpu')

    def test_returns_cuda_when_vram_sufficient(self):
        free_bytes = int(8 * 1024 ** 3)   # 8 GB free
        total_bytes = int(8 * 1024 ** 3)  # 8 GB total
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.mem_get_info', return_value=(free_bytes, total_bytes)):
            result = probe_vram(2.0)   # model needs 2 GB
        self.assertEqual(result, 'cuda')

    def test_returns_cpu_when_vram_insufficient(self):
        free_bytes = int(1 * 1024 ** 3)   # 1 GB free
        total_bytes = int(8 * 1024 ** 3)  # 8 GB total
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.mem_get_info', return_value=(free_bytes, total_bytes)):
            result = probe_vram(4.0)   # model needs 4 GB
        self.assertEqual(result, 'cpu')

    def test_forced_cpu_mode(self):
        os.environ['VISIONE_OFFLOAD_MODE'] = 'cpu'
        with patch('torch.cuda.is_available', return_value=True):
            result = probe_vram(0.1)
        self.assertEqual(result, 'cpu')

    def test_forced_disk_mode(self):
        os.environ['VISIONE_OFFLOAD_MODE'] = 'disk'
        result = probe_vram(0.1)
        self.assertEqual(result, 'disk')

    def test_vram_threshold_respected(self):
        os.environ['VISIONE_VRAM_THRESHOLD'] = '0.99'
        # 80% free but threshold is 99% → should NOT load on GPU
        free_bytes = int(8 * 0.8 * 1024 ** 3)
        total_bytes = int(8 * 1024 ** 3)
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.mem_get_info', return_value=(free_bytes, total_bytes)):
            result = probe_vram(0.1)  # model is tiny
        # 80% < 99% → should not be 'cuda'
        self.assertNotEqual(result, 'cuda')

    def test_mem_get_info_failure_falls_back_to_cpu(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.mem_get_info', side_effect=RuntimeError("driver error")):
            result = probe_vram(1.0)
        self.assertEqual(result, 'cpu')


# ---------------------------------------------------------------------------
# Tests: suggest_batch_size
# ---------------------------------------------------------------------------

class TestSuggestBatchSize(unittest.TestCase):

    def test_cuda_keeps_requested(self):
        self.assertEqual(suggest_batch_size(16, 'cuda'), 16)

    def test_cuda_risky_keeps_requested(self):
        self.assertEqual(suggest_batch_size(16, 'cuda_risky'), 16)

    def test_cpu_halves_batch(self):
        self.assertEqual(suggest_batch_size(16, 'cpu'), 8)

    def test_cpu_minimum_one(self):
        self.assertEqual(suggest_batch_size(1, 'cpu'), 1)

    def test_disk_forces_one(self):
        self.assertEqual(suggest_batch_size(32, 'disk'), 1)


# ---------------------------------------------------------------------------
# Tests: get_offload_dir
# ---------------------------------------------------------------------------

class TestGetOffloadDir(unittest.TestCase):

    def setUp(self):
        os.environ.pop('VISIONE_OFFLOAD_DIR', None)

    def test_default_path(self):
        self.assertEqual(get_offload_dir(), '/cache/model_offload')

    def test_env_override(self):
        os.environ['VISIONE_OFFLOAD_DIR'] = '/tmp/my_offload'
        self.assertEqual(get_offload_dir(), '/tmp/my_offload')
        os.environ.pop('VISIONE_OFFLOAD_DIR')


# ---------------------------------------------------------------------------
# Tests: load_model_with_offload
# ---------------------------------------------------------------------------

class TestLoadModelWithOffload(unittest.TestCase):

    def setUp(self):
        for key in ('VISIONE_VRAM_THRESHOLD', 'VISIONE_OFFLOAD_MODE'):
            os.environ.pop(key, None)

    def test_loads_on_cpu_when_no_cuda(self):
        model = _make_tiny_model()
        with patch('torch.cuda.is_available', return_value=False):
            result = load_model_with_offload(lambda: model, model_size_gb=0.001)
        self.assertIsInstance(result, nn.Module)
        # Should stay on CPU
        self.assertEqual(str(next(result.parameters()).device), 'cpu')

    def test_cuda_oom_falls_back_to_cpu(self):
        """Simulate OOM on first .to('cuda') call → model should land on CPU."""
        os.environ['VISIONE_OFFLOAD_MODE'] = 'auto'

        call_count = [0]
        original_model = _make_tiny_model()

        def _mock_load():
            return _make_tiny_model()

        oom_error = torch.cuda.OutOfMemoryError("mocked OOM")

        # Patch probe_vram to return 'cuda' so we enter the GPU branch
        with patch('common.vram_aware_loader.probe_vram', return_value='cuda'), \
             patch.object(nn.Module, 'to', side_effect=oom_error) as mock_to, \
             patch('torch.cuda.empty_cache') as mock_empty:
            result = load_model_with_offload(_mock_load, model_size_gb=0.001)

        mock_empty.assert_called_once()
        self.assertIsInstance(result, nn.Module)

    def test_forced_cpu_mode_never_calls_cuda(self):
        os.environ['VISIONE_OFFLOAD_MODE'] = 'cpu'
        model = _make_tiny_model()
        cuda_to_calls = []

        original_to = nn.Module.to

        def _track_to(self_m, *args, **kwargs):
            if args and args[0] == 'cuda':
                cuda_to_calls.append(args[0])
            return original_to(self_m, *args, **kwargs)

        with patch('torch.cuda.is_available', return_value=True), \
             patch.object(nn.Module, 'to', _track_to):
            result = load_model_with_offload(lambda: model, model_size_gb=0.001)

        self.assertEqual(cuda_to_calls, [], "Model should never be moved to CUDA in cpu mode")


# ---------------------------------------------------------------------------
# Tests: load_hf_model_with_offload
# ---------------------------------------------------------------------------

class TestLoadHfModelWithOffload(unittest.TestCase):

    def setUp(self):
        for key in ('VISIONE_VRAM_THRESHOLD', 'VISIONE_OFFLOAD_MODE'):
            os.environ.pop(key, None)

    def test_cpu_mode_passes_cpu_device_map(self):
        os.environ['VISIONE_OFFLOAD_MODE'] = 'cpu'
        captured_kwargs = {}

        def _fake_from_pretrained(**kw):
            captured_kwargs.update(kw)
            return _make_tiny_model()

        load_hf_model_with_offload(_fake_from_pretrained, model_size_gb=0.001)

        self.assertEqual(captured_kwargs.get('device_map'), {"": "cpu"})

    def test_disk_mode_passes_offload_folder(self):
        os.environ['VISIONE_OFFLOAD_MODE'] = 'disk'
        captured_kwargs = {}

        def _fake_from_pretrained(**kw):
            captured_kwargs.update(kw)
            return _make_tiny_model()

        with patch('common.vram_aware_loader._ensure_offload_dir'):
            load_hf_model_with_offload(
                _fake_from_pretrained, model_size_gb=0.001, offload_dir='/tmp/offload_test'
            )

        self.assertEqual(captured_kwargs.get('offload_folder'), '/tmp/offload_test')
        self.assertTrue(captured_kwargs.get('offload_state_dict'))


# ---------------------------------------------------------------------------
# Tests: load_checkpoint_to_device
# ---------------------------------------------------------------------------

class TestLoadCheckpointToDevice(unittest.TestCase):

    def _make_checkpoint_bytes(self):
        """Return a dict checkpoint that matches the tiny model's state dict."""
        model = _make_tiny_model()
        return model.state_dict()

    def test_loads_state_dict_on_cpu(self):
        model = _make_tiny_model()
        state = self._make_checkpoint_bytes()

        with patch('torch.load', return_value=state):
            result = load_checkpoint_to_device(
                checkpoint_path='/fake/checkpoint.pt',
                model=model,
                strategy='cpu',
            )

        self.assertIsInstance(result, nn.Module)
        self.assertEqual(str(next(result.parameters()).device), 'cpu')

    def test_oom_during_cuda_move_falls_back(self):
        model = _make_tiny_model()
        state = self._make_checkpoint_bytes()
        oom = torch.cuda.OutOfMemoryError("mocked OOM")

        original_to = nn.Module.to

        def _raise_on_cuda(self_m, *args, **kwargs):
            if args and args[0] == 'cuda':
                raise oom
            return original_to(self_m, *args, **kwargs)

        with patch('torch.load', return_value=state), \
             patch.object(nn.Module, 'to', _raise_on_cuda), \
             patch('torch.cuda.empty_cache'):
            result = load_checkpoint_to_device(
                checkpoint_path='/fake/checkpoint.pt',
                model=model,
                strategy='cuda',
            )

        # After OOM fallback the model should still be a valid module on CPU
        self.assertIsInstance(result, nn.Module)


if __name__ == '__main__':
    unittest.main()
