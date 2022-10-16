import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")