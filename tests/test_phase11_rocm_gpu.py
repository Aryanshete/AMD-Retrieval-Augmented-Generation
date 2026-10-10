"""Tests for Phase 11: ROCm GPU Model Loader & Optimization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.models.model_loader import DeviceInfo, ModelLoader, get_device_info


class TestPhase11RocmGpu(unittest.TestCase):
    def test_device_info_inspection(self):
        """get_device_info must provide valid device diagnostics without crashing."""
        info = get_device_info()
        self.assertIsInstance(info, DeviceInfo)
        self.assertIsInstance(info.is_rocm, bool)
        self.assertIsInstance(info.is_gpu_available, bool)
        self.assertIn(info.device, ["cuda", "cpu"])
        self.assertIsInstance(info.torch_version, str)
        self.assertIsInstance(info.vram_allocated_mb, float)

        info_dict = info.to_dict()
        self.assertIn("torch_version", info_dict)
        self.assertIn("vram_allocated_mb", info_dict)

    def test_model_loader_path_resolution(self):
        """ModelLoader must resolve paths strictly from local models directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_model = temp_path / "mock_vlm"
            fake_model.mkdir()

            loader = ModelLoader(models_dir=temp_path)
            resolved = loader.resolve_model_path("mock_vlm")
            self.assertEqual(resolved, fake_model)
            self.assertTrue(resolved.exists())

    def test_model_loader_handles_missing_offline_weights_safely(self):
        """ModelLoader should gracefully report missing offline paths without crashing or network access."""
        loader = ModelLoader(models_dir=Path("/nonexistent/models"))
        res = loader.load_offline_model("nonexistent_model")
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
