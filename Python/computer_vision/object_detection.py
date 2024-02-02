#! python
    
import torch
import cv2
import ultralytics
ultralytics.checks()


model = ultralytics.YOLO('yolov8n.pt')  # initializes model

# Path to the video file or integer for webcam
video_source = 0  # Use 0 for webcam

# Initialize video capture
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    result = model.predict(frame)[0]
    
    # Extract data from the results
    detections = result.boxes
    for i in range(len(detections)):
        obj = int(detections[i].cls[0])  # class labels
        conf = float(detections[i].conf)  # confidence scores
        xyxy = torch.squeeze(detections[i].xyxy)  # bounding boxes

        
        # Draw bounding boxes and labels on the frame
        label = f'{model.names[obj]} {conf:.2f}'
        color = (obj*10 & 0xFF, (255-10*obj) & 0xFF, 0)
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(
            xyxy[2]), int(xyxy[3])), color, 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(
            xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # Show the frame
        win_title = 'YOLOv5 Object Detection'
        cv2.imshow(win_title, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty(win_title,
                                                cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
