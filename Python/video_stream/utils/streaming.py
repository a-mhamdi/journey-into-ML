import os
import sys
import cv2


class VStream:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        cap = cv2.VideoCapture()
        cap.open(0)  # rtsp or ip cam
        self.capture = cap

    def __del__(self):
        self.capture.release()

    def update(self):
        """
        Update the captured frame and return the resized frame.

        Parameters:
            self (object): The object instance
        Returns:
            numpy.ndarray: The resized frame
        """
        status, frame = self.capture.read()
        if status:
            resized_frame = cv2.resize(frame, (self.width, self.height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            rx, ry = self.width/1944, self.height/2592
            return frame, resized_frame, (rx, ry)
        else:
            raise Exception('Could not read frame')
