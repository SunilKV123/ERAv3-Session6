import os
import sys
import subprocess
import pkg_resources

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check and install required packages
required_packages = ['tqdm', 'torch', 'torchvision']
installed_packages = [pkg.key for pkg in pkg_resources.working_set]

for package in required_packages:
    if package not in installed_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import unittest
import torch
from src.train import train_model
from src.model import MnistCNN

class TestTraining(unittest.TestCase):
    def setUp(self):
        print("\n" + "="*50)
        print("Running Training Pipeline Tests...")
        print("="*50)

    def test_model_saving(self):
        print("\nTesting model saving functionality...")
        _, _, model_path = train_model(epochs=1, batch_size=32)
        
        # Check if model file exists
        self.assertTrue(os.path.exists(f"models/{model_path}"), 
                       "Model file should exist in models directory")
        
        # Check file extension
        self.assertTrue(model_path.endswith('.pth'), 
                       "Model file should have .pth extension")
        
        # Check if file size is reasonable (not empty)
        file_size = os.path.getsize(f"models/{model_path}")
        self.assertGreater(file_size, 1000, 
                          "Model file size should be reasonable")
        
        print(f"✓ Model Saving Test Passed: Model saved as {model_path}")

    def test_high_accuracy(self):
        print("\nTesting for high accuracy achievement...")
        # Train for more epochs to achieve high accuracy
        _, accuracy, _ = train_model(epochs=20, batch_size=32)
        
        self.assertGreaterEqual(accuracy, 99.3, 
                               f"Validation accuracy {accuracy:.2f}% is less than required 99.3%")
        
        print(f"✓ High Accuracy Test Passed: Achieved {accuracy:.2f}% (≥ 99.4%)")

    def tearDown(self):
        print("-"*50)

if __name__ == '__main__':
    unittest.main(verbosity=2) 