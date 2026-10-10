"""Tests for Phase 13: Container & Build Verification (Sections 2, 22, 31, 32).

Ensures:
- Exact mandatory ROCm base image is preserved.
- No HTTP ports exposed (Unix domain socket only).
- ROCm PyTorch safety (requirements.txt does not overwrite torch).
- Working directory, directory creation, and CMD contract match specification.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest


class TestPhase13DockerfileAndEnv(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.dockerfile = self.repo_root / "Dockerfile"
        self.req_file = self.repo_root / "app" / "requirements.txt"

    def test_dockerfile_mandatory_base_image(self):
        """Dockerfile must use the exact AMD-mandated ROCm base image without modification."""
        self.assertTrue(self.dockerfile.exists(), "Dockerfile must exist at repository root")
        content = self.dockerfile.read_text(encoding="utf-8")

        expected_base = "rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
        m = re.search(r"FROM\s+(\S+)", content)
        self.assertIsNotNone(m, "FROM instruction not found in Dockerfile")
        base_image = m.group(1).strip()
        self.assertEqual(
            base_image,
            expected_base,
            f"Mandatory ROCm base image changed! Expected {expected_base}, found {base_image}",
        )

    def test_dockerfile_no_exposed_http_port(self):
        """Evaluation contract strictly prohibits exposing any HTTP port."""
        content = self.dockerfile.read_text(encoding="utf-8")
        # Check that EXPOSE is not in Dockerfile
        expose_lines = [line for line in content.splitlines() if line.strip().startswith("EXPOSE")]
        self.assertEqual(len(expose_lines), 0, f"Found prohibited EXPOSE directive: {expose_lines}")

    def test_dockerfile_cmd_worker_lifecycle(self):
        """Dockerfile CMD must start worker.py and keep container alive (Section 25)."""
        content = self.dockerfile.read_text(encoding="utf-8")
        self.assertIn("worker.py", content)
        self.assertIn("sleep infinity", content)

    def test_requirements_preserves_rocm_torch(self):
        """Requirements must not contain torch/torchvision/torchaudio to prevent replacing ROCm torch."""
        self.assertTrue(self.req_file.exists(), "app/requirements.txt must exist")
        reqs = self.req_file.read_text(encoding="utf-8").splitlines()
        for r in reqs:
            pkg = r.strip().lower()
            if not pkg or pkg.startswith("#"):
                continue
            pkg_name = re.split(r"[><=~]", pkg)[0].strip()
            self.assertNotIn(
                pkg_name,
                ["torch", "torchvision", "torchaudio"],
                f"Hazardous dependency '{pkg_name}' found! Installing torch via pip overwrites ROCm build.",
            )

    def test_required_mount_directories_created(self):
        """Dockerfile must create required directories: corpus, output, index, run, models."""
        content = self.dockerfile.read_text(encoding="utf-8")
        for d in ["/app/corpus", "/app/output", "/app/index", "/app/run", "/models"]:
            self.assertIn(d, content)


if __name__ == "__main__":
    unittest.main()
