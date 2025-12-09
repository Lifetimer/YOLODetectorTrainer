from ultralytics import YOLO
import cv2


def main(model_path: str):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 90)
    print("按 'q' 键退出程序")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break
        
        results = model(frame, conf = 0.5)
        annotated_frame = results[0].plot()
        
        cv2.imshow('YOLOv11 Real-time Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(model_path='best.pt')