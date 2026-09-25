import unittest
import io
import torch
from PIL import Image
from model import SimpleImageClassifier, ImageClassificationService

class TestImageClassifierModel(unittest.TestCase):
    def setUp(self):
        self.model = SimpleImageClassifier(num_classes=10)
        self.model.eval()

    def test_model_architecture_output_shape(self):
        """Ensure batch tensor outputs shape (batch_size, num_classes)."""
        dummy_batch = torch.randn(4, 3, 32, 32)
        output = self.model(dummy_batch)
        self.assertEqual(output.shape, (4, 10))

    def test_class_labels_count(self):
        """Ensure class label registry has exactly 10 classes."""
        self.assertEqual(len(self.model.classes), 10)


class TestImageClassificationService(unittest.TestCase):
    def setUp(self):
        self.service = ImageClassificationService()

    def _create_sample_image(self, width=64, height=64, color="blue"):
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_preprocess_tensor_shape(self):
        """Test that arbitrary image sizes are resized to standard (1, 3, 32, 32)."""
        sample_img = self._create_sample_image(width=128, height=80, color="green")
        tensor = self.service.preprocess(sample_img)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))

    def test_predict_structure(self):
        """Test that the prediction returns required dictionary keys."""
        sample_img = self._create_sample_image(width=32, height=32, color="red")
        result = self.service.predict(sample_img)
        
        self.assertIn("class_index", result)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("all_probabilities", result)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(len(result["all_probabilities"]), 10)

    def test_invalid_input_type(self):
        """Verify proper error handling when unsupported type is passed."""
        with self.assertRaises(ValueError):
            self.service.preprocess(12345)


if __name__ == "__main__":
    print("Running Image Classification Test Suite...")
    suite = unittest.TestLoader().loadTestsFromNames([
        'test.TestImageClassifierModel',
        'test.TestImageClassificationService'
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        exit(1)