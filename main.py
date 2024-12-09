import cv2
from ultralytics import YOLO
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import time


shared_box = None
lock = threading.Lock()
running = True
active_predictions = 0
max_predictions = 2
thread_pool = ThreadPoolExecutor(max_workers=2)

active_detections = []
detections_lock = threading.Lock()

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


model = YOLO('yolov8x.pt')
model.conf = 0.8  
cap = cv2.VideoCapture(0)


# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

def cam_runner():
    global running, active_predictions
    last_predict_time = time.time()
    frame_count = 0
    
    while running:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        frame_count += 1

        
        if frame_count % 10 == 0 and active_predictions < max_predictions:
            with lock:
                active_predictions += 1
            thread_pool.submit(predict, frame.copy())
            last_predict_time = current_time
            
      
        display_frame = draw_detections(frame.copy())
        cv2.imshow('frame', display_frame)
        
       
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            running = False
            break

def predict(frame):
    global active_predictions
    if not running:
        return
        
    try:
        results = model(frame, imgsz=320, verbose=False)
        new_detections = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls in VEHICLE_CLASSES and conf >= 0.8:
                    xyxy = box.xyxy[0].cpu().numpy()
                    label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
                    new_detections.append((xyxy, label))
        
        
        with detections_lock:
            active_detections.clear()
            active_detections.extend(new_detections)
            
    except Exception as e:
        print(f"Error in predict: {e}")
    finally:
        with lock:
            active_predictions -= 1

def draw_detections(frame):
    with detections_lock:
        for det in active_detections:
            xyxy, label = det
            cv2.rectangle(frame, 
                         (int(xyxy[0]), int(xyxy[1])), 
                         (int(xyxy[2]), int(xyxy[3])), 
                         (0, 255, 0), 2)
            cv2.putText(frame, label, 
                       (int(xyxy[0]), int(xyxy[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
    return frame

def detect_from_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    
    output_path = 'output_' + video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:  
            try:
                results = model(frame, imgsz=320, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if cls in VEHICLE_CLASSES and conf >= 0.8:
                            xyxy = box.xyxy[0].cpu().numpy()
                            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
                            
                            cv2.rectangle(frame, 
                                        (int(xyxy[0]), int(xyxy[1])), 
                                        (int(xyxy[2]), int(xyxy[3])), 
                                        (0, 255, 0), 2)
                            cv2.putText(frame, label, 
                                      (int(xyxy[0]), int(xyxy[1]-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                      (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing frame: {e}")
                
        out.write(frame)
        cv2.imshow('Video Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    try:

        results = model(image, imgsz=320, verbose=False)
        

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls in VEHICLE_CLASSES and conf >= 0.8:
                    xyxy = box.xyxy[0].cpu().numpy()
                    label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
                    

                    cv2.rectangle(image, 
                                (int(xyxy[0]), int(xyxy[1])), 
                                (int(xyxy[2]), int(xyxy[3])), 
                                (0, 255, 0), 2)
                    cv2.putText(image, label, 
                              (int(xyxy[0]), int(xyxy[1]-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (0, 255, 0), 2)
        
        output_path = 'detected_' + image_path.split('/')[-1]
        cv2.imwrite(output_path, image)
        cv2.imshow('Image Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        cmd = input("Enter 'v' for video detection, 'i' for image detection, or 'c' for camera detection: ")
        if cmd == 'v':
            video_path = input("Enter video path: ")
            detect_from_video(video_path)
        elif cmd == 'i':
            image_path = input("Enter image path: ")
            detect_from_image(image_path)
        elif cmd == 'c':
            cam_runner()
        else:
            print("Invalid command")
    except KeyboardInterrupt:
        running = False
    finally:
        thread_pool.shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()

