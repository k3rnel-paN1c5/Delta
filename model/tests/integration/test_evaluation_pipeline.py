import unittest
import torch
import os
import shutil
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from models.student_model import StudentDepthModel
from models.teacher_model import TeacherWrapper
from datasets.data_loader import UnlabeledImageDataset 
from utils.metrics import compute_depth_metrics
from torch.utils.data import DataLoader
from torchvision import transforms

class TestEvaluationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory, a fake dataset, and a fake model checkpoint."""
        cls.temp_dir = "temp_eval_test"
        cls.dataset_path = os.path.join(cls.temp_dir, "dataset")
        cls.model_path = os.path.join(cls.temp_dir, "student_model.pth")
        
        os.makedirs(cls.dataset_path, exist_ok=True)
        
        cls.input_size = 384
        
        # --- Create a fake dataset ---
        for i in range(4): # Create 4 dummy images
            random_image = (np.random.rand(cls.input_size, cls.input_size, 3) * 255).astype(np.uint8)
            Image.fromarray(random_image).save(os.path.join(cls.dataset_path, f"img_{i}.png"))
            
        # --- Create a fake model checkpoint ---
        temp_student_model = StudentDepthModel(pretrained=False)
        torch.save(temp_student_model.state_dict(), cls.model_path)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory after all tests are done."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @patch('models.teacher_model.AutoModelForDepthEstimation.from_pretrained')
    def test_evaluation_components_integration(self, mock_from_pretrained):
        """
        Tests the integration of model loading, data processing, and metric computation.
        """
        # --- 1. Mock the Teacher Model Download ---
        mock_teacher_model = MagicMock()
        mock_outputs = MagicMock()
        
        batch_size = 2
        patch_size = 14
        h_grid = self.input_size // patch_size
        w_grid = self.input_size // patch_size
        seq_len = h_grid * w_grid + 1
        
        # The mock model's output has the attributes the TeacherWrapper expects
        mock_outputs.predicted_depth = torch.randn(batch_size, self.input_size, self.input_size)
        mock_outputs.hidden_states = [torch.randn(batch_size, seq_len, 768) for _ in range(12)]
        
        # Mock teacher to return a predictable depth map shape and empty features
        mock_teacher_model.return_value = mock_outputs
        mock_teacher_model.config.patch_size = 14 # Needed for internal logic
        mock_from_pretrained.return_value = mock_teacher_model

        # --- 2. Setup Real Components ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the student model from our fake checkpoint
        student_model = StudentDepthModel(pretrained=False).to(device)
        student_model.load_state_dict(torch.load(self.model_path, map_location=device))
        student_model.eval()

        teacher_model = TeacherWrapper().to(device) # This will use the mock
        
        # Setup dataset and dataloader
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        eval_dataset = UnlabeledImageDataset(root_dir=self.dataset_path, transform=eval_transform)
        eval_dataloader = DataLoader(eval_dataset, batch_size=2)
        
        # --- 3. Run a Manual Evaluation Loop ---
        total_metrics = {
            'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0,
            'rmse_log': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0
        }
        num_samples = 0

        with torch.no_grad():
            for images in eval_dataloader:
                images = images.to(device)

                # Get predictions
                teacher_depth, _ = teacher_model(images)
                student_depth, _ = student_model(images)

                # Compute metrics for the batch
                metrics = compute_depth_metrics(student_depth, teacher_depth)
                
                # Accumulate metrics
                batch_size = images.size(0)
                for key in total_metrics:
                    total_metrics[key] += metrics[key] * batch_size
                num_samples += batch_size

        # --- 4. Assertions ---
        self.assertGreater(num_samples, 0, "The evaluation loop did not process any samples.")
        
        # Calculate final average metrics
        avg_metrics = {key: total / num_samples for key, total in total_metrics.items()}
        
        # Check that all metrics are valid numbers (not NaN or infinity)
        for key, value in avg_metrics.items():
            self.assertIsInstance(value, float)
            self.assertFalse(np.isnan(value), f"Metric '{key}' is NaN")
            self.assertFalse(np.isinf(value), f"Metric '{key}' is Infinity")