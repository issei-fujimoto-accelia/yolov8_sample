import cv2

# from models.yolo import YOLOv8Seg
from models.yoloSeg import YOLOSeg
import argparse

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize YOLOv5 Instance Segmentator
    model_path = "./weights/yolov8s-seg.onnx"
    yoloseg = YOLOSeg(model_path, conf_threshold=args.conf, iou_threshold=args.iou)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break
        
        # Update object localizer
        boxes, scores, class_ids, masks = yoloseg(frame)        
        # combined_img = yoloseg.draw_masks(frame)
        combined_img = yoloseg.draw_masks_only(frame)
        cv2.imshow("Detected Objects", combined_img)

        # if len(boxes) > 0:            
            #yoloseg.draw_masks(frame)
            # yoloseg.draw_and_visualize(frame, boxes, segments, vis=False, save=False, video=True)
        # cv2.imshow("video", frame)
        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()