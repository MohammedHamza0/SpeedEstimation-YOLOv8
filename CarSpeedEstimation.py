from ultralytics.solutions import speed_estimation
from ultralytics import YOLO
import cv2
import os

os.chdir(r"F:\YOLO Projects\Car Tracking")

# Load the YOLO model
model = YOLO(model="yolov8l.pt")
classes = model.model.names

line_pts = [(389, 475), (827, 475)]
left_counts = {'car': 0, 'bus': 0, 'truck': 0}
right_counts = {'car': 0, 'bus': 0, 'truck': 0}

# Define the target labels for counting
targetLabel = ['car', 'bus', 'truck']

# Initialize the SpeedEstimator object
speed_obj = speed_estimation.SpeedEstimator(names=classes, reg_pts=line_pts)

# Load the video file
cap = cv2.VideoCapture("highway.mp4")

# Dictionary to keep track of vehicles by their IDs
vehicle_tracker = {}


def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Background rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
    # Border rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("The frames have finished.")
        break
    
    frame = cv2.resize(frame, (1100, 700))
    
    # Get the tracks of objects
    results = model.track(frame, persist=True, show=False)
    
    # Estimate speed
    speed_obj.estimate_speed(frame, results)
    
    
    for result in results:
        
        for track in result.boxes:
            
            class_id = int(track.cls)
            bbox = track.xyxy[0].cpu().numpy()  
            conf = float(track.conf)  
            
            xmin, ymin, xmax, ymax = map(int, bbox)  
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            
        
            track_id = int(track.id) if track.id is not None else None
            
            
            if track_id is not None and track_id not in vehicle_tracker:
                vehicle_tracker[track_id] = {'class': classes[class_id], 'counted': False}
            
            
            if track_id is not None and vehicle_tracker[track_id]['class'] in targetLabel:
                
                if not vehicle_tracker[track_id]['counted']:
                    if line_pts[0][1] - 5 < center_y < line_pts[0][1] + 5:
                        if center_x < (line_pts[0][0] + line_pts[1][0]) // 2:
                            left_counts[vehicle_tracker[track_id]['class']] += 1
                        else:
                            right_counts[vehicle_tracker[track_id]['class']] += 1
                        vehicle_tracker[track_id]['counted'] = True
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
                cv2.circle(frame, (center_x, center_y), 5, [0, 0, 255], -1)

    draw_text_with_background(frame, 
                              f"GoingDown - Cars: {left_counts['car']}, Buses: {left_counts['bus']}, Trucks: {left_counts['truck']}", 
                              (10, 30), 
                              cv2.FONT_HERSHEY_COMPLEX, 
                              0.8, 
                              (255, 255, 255),  
                              (0, 0, 0), 
                              (72, 61, 139))  
    
    
    
    draw_text_with_background(frame, 
                              f"GoingUp - Cars: {right_counts['car']}, Buses: {right_counts['bus']}, Trucks: {right_counts['truck']}", 
                              (10, 90), 
                              cv2.FONT_HERSHEY_COMPLEX, 
                              0.8, 
                              (255, 255, 255),  
                              (0, 0, 0),  
                              (72, 61, 139))  

    cv2.imshow('Vehicle Tracking', frame)
    
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
