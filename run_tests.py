import unittest
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite) 