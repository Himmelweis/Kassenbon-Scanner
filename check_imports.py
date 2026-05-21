import sys
print(sys.executable)

import torch
print("torch ok", torch.__version__)

from paddleocr import PaddleOCR
print("paddleocr ok")