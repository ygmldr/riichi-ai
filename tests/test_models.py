import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jiangpu.models import load_bundle, CHECKPOINT_FIXTURE
        cls.bundle = load_bundle(CHECKPOINT_FIXTURE, device="cpu")

    def test_pin_identity(self):
        from jiangpu.models import checkpoint_sha256, CHECKPOINT_FIXTURE
        got = checkpoint_sha256(CHECKPOINT_FIXTURE)
        self.assertEqual(len(got), 64)
        self.assertTrue(got.startswith("0a44613cb9c7edae1057"), got)

    def test_steps_and_version(self):
        self.assertEqual(self.bundle.steps, 396000)
        self.assertEqual(self.bundle.version, 4)
        self.assertIsNotNone(self.bundle.aux_net)
        self.assertIsNotNone(self.bundle.grp)

    def test_forward_shapes(self):
        import torch
        obs = torch.zeros(2, 1012, 34, dtype=torch.float32)
        mask = torch.ones(2, 46, dtype=torch.bool)
        with torch.inference_mode():
            phi = self.bundle.brain(obs, events=None)
            q, _v = self.bundle.dqn(phi, mask)
            rank_logits, = self.bundle.aux_net(phi)
        self.assertEqual(phi.shape, (2, 1024))
        self.assertEqual(q.shape, (2, 46))
        self.assertEqual(rank_logits.shape, (2, 4))

    def test_cudnn_benchmark_disabled_for_determinism(self):
        import torch
        self.assertFalse(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()
