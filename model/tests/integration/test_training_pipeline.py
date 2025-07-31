import unittest
from unittest.mock import MagicMock
import torch
import os
import shutil
import numpy as np
from PIL import Image

from config import config
from models.student_model import StudentDepthModel
from models.teacher_model import TeacherWrapper
from models.factory import ModelFactory
from criterions.criterion import DistillationLoss
from criterions.factory import CriterionFactory
from utils.factory import OptimizerFactory
from datasets.data_loader import UnlabeledImageDataset
from torch.utils.data import DataLoader
from unittest.mock import patch

class TestTrainingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory with a fake dataset for all tests in this class."""
        cls.temp_dir = "temp_test_data"
        cls.image_dir = os.path.join(cls.temp_dir, "images")
        
        os.makedirs(cls.image_dir, exist_ok=True)
        
        # Create a few dummy images and depth maps
        for i in range(4):
            # Create a random RGB image
            random_image = (np.random.rand(384, 384, 3) * 255).astype(np.uint8)
            Image.fromarray(random_image).save(os.path.join(cls.image_dir, f"img_{i}.png"))
            

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory after all tests are done."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @patch('models.teacher_model.AutoModelForDepthEstimation.from_pretrained')
    def test_full_training_step(self, mock_from_pretrained):
        """
        Tests if a full training step (data -> model -> loss -> backward) can execute.
        """
        # --- 1. Mock the Teacher Model Download ---
        # We still mock the teacher to avoid downloading large weights during testing.
        # We configure its output to be realistic.
        mock_teacher_model = MagicMock()
        mock_outputs = MagicMock()
        # Mock teacher output shape for a given input size
        mock_outputs.predicted_depth = torch.randn(2, 384, 384) 
        # Mock hidden states with realistic shapes
        mock_outputs.hidden_states = [torch.randn(2, 730, 768) for _ in range(12)] # Assuming 384x384 input, patch 14
        mock_teacher_model.return_value = mock_outputs
        mock_teacher_model.config.patch_size = 14
        mock_from_pretrained.return_value = mock_teacher_model

        # --- 2. Setup Real Components ---
        device = config.DEVICE
        
        student_model = ModelFactory.create_student_model(config).to(device)
        teacher_model = ModelFactory.create_teacher_model(config).to(device)
        criterion = CriterionFactory.create_criterion(config).to(device)
        optimizer = OptimizerFactory.create_optimizer(student_model, config)
    

        # Create a dataset and dataloader with our fake data
        # We pass a simple transform that just converts to a tensor
        transform = lambda x: torch.from_numpy(np.array(x)).float().permute(2, 0, 1) / 255.0 if x.mode == 'RGB' else torch.from_numpy(np.array(x)).float().unsqueeze(0)
        
        dataset = UnlabeledImageDataset(
            root_dir=self.image_dir,
            transform=transform
        )
        data_loader = DataLoader(dataset, batch_size=2)
        
        # --- 3. Run a Single Training Step ---
        # Fetch one batch of data
        images = next(iter(data_loader))
        images = images.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward passes
        student_depth, student_features = student_model(images)
        teacher_depth, teacher_features = teacher_model(images)
        
        # Loss calculation
        loss = criterion(student_depth, teacher_depth, None, None, student_features, teacher_features)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # --- 4. Assertions ---
        # Did we get a valid loss value?
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0) # Check if it's a scalar
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
        
        # Did the model's weights get gradients?
        # This confirms that the backward pass was successful.
        grad_sum = sum(p.grad.sum() for p in student_model.parameters() if p.grad is not None)
        self.assertNotEqual(grad_sum.item(), 0)