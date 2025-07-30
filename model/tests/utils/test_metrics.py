import unittest
import torch
import math

from utils.metrics import compute_depth_metrics


class TestDepthMetrics(unittest.TestCase):

    def test_perfect_prediction(self):
        """Test metrics when prediction is identical to the target."""
        target = torch.tensor([[[[10., 20.], [30., 40.]]]])
        pred = target.clone()
        
        metrics = compute_depth_metrics(pred, target)
        
        self.assertAlmostEqual(metrics['abs_rel'], 0.0)
        self.assertAlmostEqual(metrics['sq_rel'], 0.0)
        self.assertAlmostEqual(metrics['rmse'], 0.0)
        self.assertAlmostEqual(metrics['rmse_log'], 0.0)
        self.assertAlmostEqual(metrics['a1'], 1.0)
        self.assertAlmostEqual(metrics['a2'], 1.0)
        self.assertAlmostEqual(metrics['a3'], 1.0)

    def test_simple_known_error(self):
        """Test metrics with a simple, predictable error."""
        target = torch.full((1, 1, 2, 2), 2.0)  # All target depths are 2.0
        pred = torch.full((1, 1, 2, 2), 2.5)   # All predicted depths are 2.5

        metrics = compute_depth_metrics(pred, target)

        # Manually calculated expected values
        expected_abs_rel = 0.25  # abs(2.5 - 2) / 2
        expected_sq_rel = 0.125  # ((2.5 - 2)**2) / 2
        expected_rmse = 0.5      # sqrt((2.5 - 2)**2)
        expected_rmse_log = abs(math.log(2.5) - math.log(2.0))
        
        # Thresh = max(2.5/2, 2/2.5) = 1.25
        # 1.25 < 1.25 is False
        expected_a1 = 0.0
        # 1.25 < 1.25**2 (1.5625) is True
        expected_a2 = 1.0
        # 1.25 < 1.25**3 (1.953) is True
        expected_a3 = 1.0

        self.assertAlmostEqual(metrics['abs_rel'], expected_abs_rel)
        self.assertAlmostEqual(metrics['sq_rel'], expected_sq_rel)
        self.assertAlmostEqual(metrics['rmse'], expected_rmse)
        self.assertAlmostEqual(metrics['rmse_log'], expected_rmse_log)
        self.assertAlmostEqual(metrics['a1'], expected_a1)
        self.assertAlmostEqual(metrics['a2'], expected_a2)
        self.assertAlmostEqual(metrics['a3'], expected_a3)

    def test_with_invalid_mask(self):
        """Test that invalid values (e.g., 0) are correctly ignored."""
        target = torch.tensor([[[[10., 0.], [30., 0.]]]])
        pred = torch.tensor([[[[5., 100.], [15., 100.]]]])
        
        metrics = compute_depth_metrics(pred, target)

        # The metrics should be calculated only on the valid pixels:
        # target_valid = [10., 30.], pred_valid = [5., 15.]
        expected_abs_rel = 0.5  # mean(abs(5-10)/10, abs(15-30)/30) = mean(0.5, 0.5)

        expected_sq_rel = 5.0 # mean(25/10, 225/30) = mean(2.5, 7.5) = 5.0

        expected_rmse = math.sqrt( ((5-10)**2 + (15-30)**2) / 2. ) # sqrt((25+225)/2) = sqrt(125)
        expected_rmse_log = math.sqrt( ((math.log(5)-math.log(10))**2 + (math.log(15)-math.log(30))**2) / 2. )
        # This is sqrt((log(0.5)**2 + log(0.5)**2)/2) = abs(log(0.5))
        
        # Thresh for both valid points is max(10/5, 5/10) = 2.0
        expected_a1 = 0.0 # 2.0 is not < 1.25
        expected_a2 = 0.0 # 2.0 is not < 1.5625
        expected_a3 = 0.0 # 2.0 is not < 1.953

        self.assertAlmostEqual(metrics['abs_rel'], expected_abs_rel)
        self.assertAlmostEqual(metrics['sq_rel'], expected_sq_rel)
        self.assertAlmostEqual(metrics['rmse'], expected_rmse, places=6)
        self.assertAlmostEqual(metrics['rmse_log'], abs(math.log(0.5)))
        self.assertAlmostEqual(metrics['a1'], expected_a1)
        self.assertAlmostEqual(metrics['a2'], expected_a2)
        self.assertAlmostEqual(metrics['a3'], expected_a3)

    def test_no_valid_data(self):
        """Test the case where all target values are zero, resulting in no valid data."""
        target = torch.zeros((1, 1, 5, 5))
        pred = torch.ones((1, 1, 5, 5))

        metrics = compute_depth_metrics(pred, target)
        
        # Check that all metrics are NaN as no valid pixels exist for comparison
        self.assertTrue(math.isnan(metrics['abs_rel']))
        self.assertTrue(math.isnan(metrics['sq_rel']))
        self.assertTrue(math.isnan(metrics['rmse']))
        self.assertTrue(math.isnan(metrics['rmse_log']))
        self.assertTrue(math.isnan(metrics['a1']))
        self.assertTrue(math.isnan(metrics['a2']))
        self.assertTrue(math.isnan(metrics['a3']))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)