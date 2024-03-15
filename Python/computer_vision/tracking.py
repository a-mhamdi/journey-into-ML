#! python

from collections import defaultdict

import numpy as np

import cv2
import ultralytics
ultralytics.checks()

model = ultralytics.YOLO('yolov8n.pt')

# Store the track history
track_history = defaultdict(lambda: [])

# Path to the video file or integer for webcam
video_source = 0  # Use 0 for webcam

# Initialize video capture
cap = cv2.VideoCapture(video_source)


win_title = 'YOLOv8 Object Detection'
while cap.isOpened():
	try:
		ret, frame = cap.read()
		if not ret:
			break

		# Run YOLOv8 tracking on the frame, persisting tracks between frames
		results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml")

		# Get the boxes and track IDs
		boxes = results[0].boxes.xywh.cpu()
		track_ids = results[0].boxes.id.int().cpu().tolist()

		# Visualize the results on the frame
		annotated_frame = results[0].plot()

		# Plot the tracks
		for box, track_id in zip(boxes, track_ids):
			x, y, w, h = box
			track = track_history[track_id]
			track.append((float(x), float(y)))  # x, y center point
			if len(track) > 30:  # retain 90 tracks for 90 frames
				track.pop(0)

		# Draw the tracking lines
		points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(annotated_frame, [points], isClosed=False, color=(20, 0, 250), thickness=5)

		# Show the frame
		cv2.imshow(win_title, annotated_frame)
	except Exception as e:
		print("Error")

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q") or cv2.getWindowProperty(win_title,
					                    cv2.WND_PROP_VISIBLE) < 1:
		break

# Release resources
cap.release()
cv2.destroyAllWindows()
