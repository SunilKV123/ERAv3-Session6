import os
import sys
import subprocess
import pkg_resources

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check and install required packages
required_packages = ['torch', 'torchvision']
installed_packages = [pkg.key for pkg in pkg_resources.working_set]

for package in required_packages:
    if package not in installed_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import unittest
import torch
import torch.nn as nn
from src.model import MnistCNN
from src.utils import count_parameters

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MnistCNN()
        print("\n" + "="*50)
        print("Running Model Architecture Tests...")
        print("="*50)

    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 20000, 
                       f"Model has {param_count} parameters, should be less than 20000")
        print(f"✓ Parameter Count Test Passed: Model has {param_count} parameters (< 20000)")

    def test_batch_normalization(self):
        has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) 
                           for m in self.model.modules())
        self.assertTrue(has_batchnorm, 
                       "Model should use Batch Normalization layers")
        
        # Count number of batch norm layers
        batchnorm_count = sum(1 for m in self.model.modules() 
                             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
        print(f"✓ Batch Normalization Test Passed: Found {batchnorm_count} BatchNorm layers")

    def test_dropout(self):
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())
        self.assertTrue(has_dropout, 
                       "Model should use Dropout layers")
        
        # Check dropout rate
        dropout_layers = [m for m in self.model.modules() if isinstance(m, nn.Dropout)]
        for layer in dropout_layers:
            self.assertGreater(layer.p, 0.0, "Dropout rate should be greater than 0")
            self.assertLess(layer.p, 0.5, "Dropout rate should be less than 0.5")
        
        print(f"✓ Dropout Test Passed: Found {len(dropout_layers)} Dropout layers")

    def test_global_average_pooling(self):
        # Check for either AdaptiveAvgPool2d or AvgPool2d
        has_global_pool = any(isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) 
                            for m in self.model.modules())
        self.assertTrue(has_global_pool, 
                       "Model should use Global Average Pooling (either AdaptiveAvgPool2d or AvgPool2d)")
        
        # Test the output with a sample input
        test_input = torch.randn(1, 12, 5, 5)  # Adjust channels and size according to your model
        for module in self.model.modules():
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                output = module(test_input)
                self.assertEqual(output.shape[-2:], (1, 1), 
                               "Global Average Pooling should reduce spatial dimensions to 1x1")
                break
        
        print("✓ Global Average Pooling Test Passed: Found correct Global Average Pooling layer")

    def tearDown(self):
        print("-"*50)

if __name__ == '__main__':
    unittest.main(verbosity=2) 