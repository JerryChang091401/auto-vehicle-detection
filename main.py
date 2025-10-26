from speed_det import SpeedUpDetector
import cv2

cap = cv2.VideoCapture("videos/driving_video_3.mp4")
detector = SpeedUpDetector(skip_frames=2, resize_width=640)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out_frame, detections, risks, fps = detector.process_frame(frame)
    
    # Draw FPS overlay
    cv2.putText(out_frame, f"FPS: {fps:.2f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.imshow("VisionDrive Demo", out_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
