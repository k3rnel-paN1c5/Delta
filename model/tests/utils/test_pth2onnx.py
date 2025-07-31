import unittest
import os
import torch
import tempfile
import shutil
from unittest.mock import patch
from io import StringIO


# Attempt to import onnx for more robust validation
try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import the function and components to be tested
from utils.pth2onnx import convert_to_onnx
from models.factory import ModelFactory
from config import config

class TestPthToOnnxConversion(unittest.TestCase):
    """
    Test suite for the pth2onnx.py conversion script.
    """

    def setUp(self):
        """
        Set up a temporary environment for each test.
        This includes creating a temporary directory and a dummy model file.
        """
        # Create a temporary directory to store test artifacts
        self.test_dir = tempfile.mkdtemp()
        
        # Define the output directory for the exported ONNX model
        self.output_dir = os.path.join(self.test_dir, 'exports')

        # Create a dummy StudentDepthModel instance
        dummy_model = ModelFactory.create_student_model(config).to(config.DEVICE)
        dummy_model.eval()

        # Save the dummy model's state dictionary to a .pth file
        self.dummy_model_path = os.path.join(self.test_dir, 'dummy_model.pth')
        torch.save(dummy_model.state_dict(), self.dummy_model_path)

    def tearDown(self):
        """
        Clean up the temporary directory and all its contents after each test.
        """
        shutil.rmtree(self.test_dir)

    def test_successful_conversion(self):
        """
        Tests the successful conversion of a .pth model to the .onnx format.
        """
        # Run the conversion function
        convert_to_onnx(self.dummy_model_path, self.output_dir, verbose=False)

        # Define the expected path for the output file
        expected_output_path = os.path.join(self.output_dir, 'dummy_model.onnx')

        # Assert that the ONNX file was created
        self.assertTrue(os.path.exists(expected_output_path), "ONNX file was not created.")
        
        # Assert that the created file is not empty
        self.assertGreater(os.path.getsize(expected_output_path), 0, "ONNX file is empty.")

        # If ONNX libraries are installed, perform a more detailed validation
        if ONNX_AVAILABLE:
            try:
                # Load the generated model
                onnx_model = onnx.load(expected_output_path)
                
                # Check the model's integrity
                onnx.checker.check_model(onnx_model)

                # Verify that the model can be loaded into an ONNX Runtime session
                ort_session = onnxruntime.InferenceSession(expected_output_path)
                self.assertIsNotNone(ort_session, "Failed to create ONNX Runtime session.")

                # Check input and output names
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name
                self.assertEqual(input_name, 'input')
                self.assertEqual(output_name, 'output')

            except Exception as e:
                self.fail(f"ONNX model validation failed: {e}")

    def test_model_file_not_found(self):
        """
        Tests that the script handles a non-existent input model file gracefully.
        """
        non_existent_path = os.path.join(self.test_dir, 'non_existent_model.pth')

        # Redirect stdout to capture the print output of the function
        with patch('sys.stdout', new=StringIO()) as fake_out:
            convert_to_onnx(non_existent_path, self.output_dir, verbose=False)
            
            # Check that the appropriate error message was printed
            output = fake_out.getvalue().strip()
            self.assertIn(f"Error: Model file not found at '{non_existent_path}'", output)
        
        # Ensure the output directory was not created since the input was invalid
        self.assertFalse(os.path.exists(self.output_dir), "Output directory should not be created on failure.")

    def test_directory_creation(self):
        """
        Tests that the output directory is created if it doesn't already exist.
        """
        # Ensure the output directory does not exist before the test
        self.assertFalse(os.path.exists(self.output_dir))
        
        # Run the conversion
        convert_to_onnx(self.dummy_model_path, self.output_dir, verbose=False)
        
        # Check that the output directory now exists
        self.assertTrue(os.path.exists(self.output_dir), "Output directory was not created.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)