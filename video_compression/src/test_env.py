# 创建test.py文件测试环境
import sys
from PyQt5.QtWidgets import QApplication, QLabel

def test_environment():
    packages = {
        'PyQt5': 'from PyQt5.QtWidgets import QLabel',
        'OpenCV': 'import cv2',
        'NumPy': 'import numpy',
        'Pygame': 'import pygame'
    }
    
    for package, test_import in packages.items():
        try:
            exec(test_import)
            print(f"✅ {package} successfully installed")
        except ImportError as e:
            print(f"❌ {package} not properly installed: {e}")

if __name__ == "__main__":
    test_environment()