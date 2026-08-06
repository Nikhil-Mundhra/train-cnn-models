"""
tests/test_model_training.py

Mock unit tests for consolidated model training pipelines (Models 1, 4, 5) using synthetic tensors.
Verifies model instantiation, tensor shapes, forward passes, loss steps, and croppers without dataset files.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "model_training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "model_training"))

from models_suite.model1_oct5k_layers.unet_layers import RetinalLayersUNet
from models_suite.model1_oct5k_layers.crop_generator import RetinaCropper
from models_suite.model4_oimhs_hole_cysts.oimhs_unet import OIMHSUNet
from models_suite.model5_oct5k_detection.detector import OCTPathologyDetector
from model_training.train_cleanup import clean_gpu_memory


class TestModel1OCT5KLayers(unittest.TestCase):
    """Mock tests for Model 1 (OCT5K 6-Retinal Layer Segmentation U-Net & Cropper)"""

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = RetinalLayersUNet(in_channels=1, num_classes=6).to(self.device)
        self.cropper = RetinaCropper(background_class_id=0)

    def test_model1_forward_and_loss(self):
        dummy_input = torch.randn(2, 1, 128, 128, device=self.device)
        dummy_target = torch.randint(0, 6, (2, 128, 128), dtype=torch.long, device=self.device)

        logits = self.model(dummy_input)
        self.assertEqual(logits.shape, (2, 6, 128, 128))

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, dummy_target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_retina_cropper(self):
        dummy_scan = torch.ones(1, 1, 100, 100)
        dummy_mask = torch.zeros(100, 100, dtype=torch.long)
        dummy_mask[30:70, 20:80] = 1  # Retinal tissue region

        # Test background masking
        masked = self.cropper.mask_background(dummy_scan, dummy_mask)
        self.assertEqual(masked[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(masked[0, 0, 50, 50].item(), 1.0)

        # Test bounding box crop
        cropped, bbox = self.cropper.crop_retina_bbox(dummy_scan[0], dummy_mask, margin=5)
        ymin, ymax, xmin, xmax = bbox
        self.assertEqual(ymin, 25)
        self.assertEqual(ymax, 74)
        self.assertEqual(xmin, 15)
        self.assertEqual(xmax, 84)
        self.assertEqual(cropped.shape, (1, ymax - ymin, xmax - xmin))


class TestModel4OIMHSHoleCysts(unittest.TestCase):
    """Mock tests for Model 4 (OIMHS Macular Hole & Intraretinal Cysts U-Net)"""

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = OIMHSUNet(in_channels=1, num_classes=5).to(self.device)

    def test_model4_forward_and_loss(self):
        dummy_input = torch.randn(2, 1, 128, 128, device=self.device)
        dummy_target = torch.randint(0, 5, (2, 128, 128), dtype=torch.long, device=self.device)

        logits = self.model(dummy_input)
        self.assertEqual(logits.shape, (2, 5, 128, 128))

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, dummy_target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class TestModel5OCT5KDetection(unittest.TestCase):
    """Mock tests for Model 5 (OCT5K Faster R-CNN Pathology Object Detector)"""

    def setUp(self):
        self.device = torch.device("cpu")
        self.detector = OCTPathologyDetector(num_classes=10).to(self.device)

    def test_model5_train_forward_and_loss(self):
        self.detector.train()

        img1 = torch.randn(3, 128, 128, device=self.device)
        img2 = torch.randn(3, 128, 128, device=self.device)

        target1 = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=self.device),
            "labels": torch.tensor([1], dtype=torch.int64, device=self.device)
        }
        target2 = {
            "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]], device=self.device),
            "labels": torch.tensor([3], dtype=torch.int64, device=self.device)
        }

        loss_dict = self.detector([img1, img2], [target1, target2])
        self.assertIn("loss_classifier", loss_dict)
        self.assertIn("loss_box_reg", loss_dict)

        total_loss = sum(loss for loss in loss_dict.values())
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

    def test_model5_eval_inference(self):
        self.detector.eval()
        img = torch.randn(3, 128, 128, device=self.device)

        with torch.no_grad():
            predictions = self.detector([img])

        self.assertEqual(len(predictions), 1)
        self.assertIn("boxes", predictions[0])
        self.assertIn("scores", predictions[0])
        self.assertIn("labels", predictions[0])


class TestTrainCleanup(unittest.TestCase):
    """Mock test for memory cleanup guard"""

    def test_clean_gpu_memory(self):
        clean_gpu_memory()


if __name__ == "__main__":
    unittest.main()
