"""
tests/test_train_convnext_cli.py

Integration tests for train_convnext.py CLI argument parsing and execution contracts.
Guarantees zero missing attribute errors, missing options, or CLI parsing crashes.
"""

import unittest
import sys
import os
import subprocess
import tempfile
import shutil
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from train import main as cli_main

class TestTrainConvNeXtCLI(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.root_dir = os.path.dirname(self.base_dir)
        self.script_path = os.path.join(self.root_dir, "train.py")
        self.temp_dir = tempfile.mkdtemp()
        
        # Build minimal synthetic dataset structure
        self.mock_dataset_root = os.path.join(self.temp_dir, "Classified")
        mock_folders = [
            "Normal (Healthy)/OCT2017",
            "Macular Degeneration Spectrum/Choroidal Neovascularization/CNV",
            "Macular Degeneration Spectrum/DRUSEN",
            "Diabetic Complications/Diabetic Macular Edema (DME)/DME"
        ]
        for rel_p in mock_folders:
            p = os.path.join(self.mock_dataset_root, rel_p)
            os.makedirs(p, exist_ok=True)
            for img_idx in range(5):
                img_file = os.path.join(p, f"test_img_{img_idx}.jpeg")
                Image.new("RGB", (64, 64), color="gray").save(img_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cli_argument_contract(self):
        """
        Verify that train_convnext.py argument parser contains ALL required contract arguments.
        Prevents AttributeError regression on missing parameters (e.g. --resume, --checkpoint-dir).
        """
        import argparse
        
        # Test argparse setup directly from script
        test_sys_argv = [
            "train_convnext.py",
            "--config", os.path.join(self.base_dir, "config", "hierarchy.yaml"),
            "--batch-size", "16",
            "--epochs-warmup", "1",
            "--epochs-finetune", "1",
            "--lr-head", "1e-4",
            "--lr-backbone", "1e-5",
            "--num-workers", "0",
            "--smoke-test",
            "--w-h1", "1.0",
            "--w-h2", "1.0",
            "--resume", os.path.join(self.temp_dir, "dummy_ckpt.pth"),
            "--checkpoint-dir", os.path.join(self.temp_dir, "checkpoints"),
            "--hf-repo", "NMundhra/OCT-Classification-Model",
            "--accum-steps", "2",
            "--save-steps", "100"
        ]
        
        orig_argv = sys.argv
        try:
            from train import parse_args
            args = parse_args(test_sys_argv[1:])
            self.assertEqual(args.batch_size, 16)
            self.assertTrue(hasattr(args, "resume"), "args must contain 'resume'")
            self.assertTrue(hasattr(args, "checkpoint_dir"), "args must contain 'checkpoint_dir'")
            self.assertTrue(hasattr(args, "hf_repo"), "args must contain 'hf_repo'")
            self.assertTrue(hasattr(args, "accum_steps"), "args must contain 'accum_steps'")
            self.assertTrue(hasattr(args, "save_steps"), "args must contain 'save_steps'")
            
            self.assertTrue(hasattr(args, "resume"), "args must contain 'resume'")
            self.assertTrue(hasattr(args, "checkpoint_dir"), "args must contain 'checkpoint_dir'")
            self.assertTrue(hasattr(args, "hf_repo"), "args must contain 'hf_repo'")
            self.assertTrue(hasattr(args, "accum_steps"), "args must contain 'accum_steps'")
            self.assertTrue(hasattr(args, "save_steps"), "args must contain 'save_steps'")
            self.assertTrue(hasattr(args, "smoke_test"), "args must contain 'smoke_test'")
            
            self.assertEqual(args.resume, os.path.join(self.temp_dir, "dummy_ckpt.pth"))
            self.assertEqual(args.checkpoint_dir, os.path.join(self.temp_dir, "checkpoints"))
        finally:
            sys.argv = orig_argv

    def test_end_to_end_smoke_test_execution(self):
        """
        Run train_convnext.py via subprocess in --smoke-test mode on synthetic dataset.
        Guarantees that main() executes end-to-end without crashing or throwing AttributeError.
        """
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OCT_DATA_ROOT"] = self.mock_dataset_root

        cmd = [
            sys.executable,
            self.script_path,
            "--config", os.path.join(self.base_dir, "config", "hierarchy.yaml"),
            "--checkpoint-dir", os.path.join(self.temp_dir, "checkpoints"),
            "--batch-size", "2",
            "--num-workers", "0",
            "--epochs-warmup", "1",
            "--epochs-finetune", "1",
            "--smoke-test"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        self.assertEqual(
            result.returncode, 0,
            f"train_convnext.py smoke test failed!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        combined_output = result.stdout + "\n" + result.stderr
        self.assertEqual(result.returncode, 0)

if __name__ == '__main__':
    unittest.main()
