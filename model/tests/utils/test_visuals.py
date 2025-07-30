import unittest
from unittest.mock import patch, call
import numpy as np

from utils.visuals import apply_color_map, plot_depth_comparison


class TestVisuals(unittest.TestCase):

    def setUp(self):
        """Set up common data for tests."""
        # A simple, predictable depth map for testing
        self.depth_map = np.array([[0, 50], [100, 255]], dtype=np.float32)
        # A depth map with all the same values
        self.flat_depth_map = np.ones((10, 10), dtype=np.float32) * 128

    def test_apply_color_map_output_shape_and_type(self):
        """
        Test if the output of apply_color_map has the correct shape and data type.
        """
        colored_map = apply_color_map(self.depth_map)
        # 1. Check if the output is a numpy array
        self.assertIsInstance(colored_map, np.ndarray)
        # 2. Check if the output shape is (H, W, 3) for RGB
        self.assertEqual(colored_map.shape, (2, 2, 3))
        # 3. Check if the data type is uint8 (values from 0-255)
        self.assertEqual(colored_map.dtype, np.uint8)
        print("\nTest `test_apply_color_map_output_shape_and_type` passed.")

    def test_apply_color_map_flat_input(self):
        """
        Test apply_color_map with a flat depth map (all values are the same).
        The function should not fail and should produce a single-color image.
        """
        try:
            colored_map = apply_color_map(self.flat_depth_map)
            # The output should be a single color, so all pixel values should be identical.
            first_pixel_color = colored_map[0, 0]
            self.assertTrue(np.all(colored_map == first_pixel_color))
            self.assertEqual(colored_map.shape, (10, 10, 3))
        except Exception as e:
            self.fail(f"apply_color_map raised an exception with flat input: {e}")
        print("Test `test_apply_color_map_flat_input` passed.")

    def test_apply_color_map_different_cmap(self):
        """
        Test if using a different colormap produces a different result.
        """
        colored_map_inferno = apply_color_map(self.depth_map, cmap='inferno')
        colored_map_viridis = apply_color_map(self.depth_map, cmap='viridis')

        # The resulting images from different colormaps should not be identical.
        self.assertFalse(np.array_equal(colored_map_inferno, colored_map_viridis))
        print("Test `test_apply_color_map_different_cmap` passed.")

    @patch('utils.visuals.plt')
    def test_plot_depth_comparison_runs_without_error(self, mock_plt):
        """
        Test if plot_depth_comparison executes without raising an error.
        We mock matplotlib to avoid showing a plot during the test.
        """
        try:
            original = np.random.rand(10, 10, 3)
            teacher = np.random.rand(10, 10)
            student = np.random.rand(10, 10)
            plot_depth_comparison(original, teacher, student, title="Test Title")
        except Exception as e:
            self.fail(f"plot_depth_comparison raised an exception: {e}")
        print("Test `test_plot_depth_comparison_runs_without_error` passed.")


    @patch('utils.visuals.plt')
    def test_plot_depth_comparison_calls(self, mock_plt):
        """
        Test if plot_depth_comparison calls the matplotlib functions with the correct arguments.
        """
        # Create dummy data
        original_img = np.zeros((10, 10, 3), dtype=np.uint8)
        teacher_depth = np.ones((10, 10), dtype=np.float32)
        student_depth = np.full((10, 10), 0.5, dtype=np.float32)
        test_title = "My Comparison"

        # Call the function to be tested
        plot_depth_comparison(original_img, teacher_depth, student_depth, title=test_title)

        # 1. Assert that a figure was created
        mock_plt.figure.assert_called_once_with(figsize=(18, 6))

        # 2. Assert that subplots were created
        self.assertEqual(mock_plt.subplot.call_count, 3)

        # 3. Assert that titles were set correctly
        expected_title_calls = [
            call("Original Image"),
            call("Teacher Depth Map"),
            call("Student Depth Map")
        ]
        mock_plt.title.assert_has_calls(expected_title_calls, any_order=False)

        # 4. Assert that `imshow` was called for each image
        self.assertEqual(mock_plt.imshow.call_count, 3)

        # 5. Assert that the main title (suptitle) was set
        mock_plt.suptitle.assert_called_once_with(test_title)

        # 6. Assert that axes were turned off for all subplots
        self.assertEqual(mock_plt.axis.call_count, 3)
        mock_plt.axis.assert_called_with("off")

        # 7. Assert that the plot was shown
        mock_plt.show.assert_called_once()
        print("Test `test_plot_depth_comparison_calls` passed.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)