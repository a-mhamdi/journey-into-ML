#! python

import os
os.environ['YOLO_VERBOSE'] = 'False'

from contextlib import redirect_stdout

import numpy as np
from collections import defaultdict

import cv2
import ultralytics

ultralytics.checks()

model = ultralytics.YOLO('models/yolov9e.pt', verbose=False)

# Set the classes to detect only persons
model.classes = [0]  # 0 represents the 'person' class

def logTrack(func):
    def wrapper(*args, **kwargs):
        with open('log/stdout.txt', 'a') as f:
            with redirect_stdout(f):
                return func(*args, **kwargs)
    return wrapper

@logTrack
def track(video_frame, persist=True):
    """
    Run YOLOv9 tracking on the frame, persisting tracks between frames

    Args:
        frame: The input frame for tracking
        persist: Boolean flag to indicate whether to persist tracks between frames

    Returns:
        annotated_frame: The frame with visualized tracking results
    """
    print('DOES IT LOG DATA?')
    
    # Run YOLOv9 tracking on the frame, persisting tracks between frames
    results = model.track(video_frame, classes=[0])

    # Get the boxes and track IDs
    boxes = results[0].boxes.xyxy
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    return boxes
