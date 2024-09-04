import cv2
import threading
import queue
import time
import numpy as np
# from models.yolo import YOLOv8Seg
from models.yoloSeg import YOLOSeg
import argparse


COLOR_SET=dict(
    green=(0, 255, 0),
    red=(0, 0, 255),
    blue=(255, 0, 0),
    pink=(255, 0, 165),
    puple=(128, 0, 128),
)
COLORS = dict(
    small=COLOR_SET["blue"],
    midium=COLOR_SET["red"],
    large=COLOR_SET["puple"],
)

class InferenceWorker(threading.Thread):
    def __init__(self, model, input_queue, output_queue, hide_bg=False, show_mask=False, show_dot=False):
        threading.Thread.__init__(self)
        self.model = model
        self.hide_bg = hide_bg
        self.show_mask = show_mask
        self.show_dot = show_dot

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = threading.Event()

        self.mask_alpha=1.0

    def run(self):
        while not self.stop_event.is_set():
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                raw_frame = frame.copy()
                boxes, scores, class_ids, masks = self.model(raw_frame)

                if self.hide_bg:
                    (hight, width, _) = frame.shape
                    blank_frame = np.full((hight, width, 3), 0, np.uint8) ## 黒
                    # blank_frame = np.full((hight, width, 3), 255, np.unit8) ## 白
                    raw_frame = blank_frame

                if self.show_mask:
                    combined_img = self.model.draw_masks_only(raw_frame, mask_alpha=self.mask_alpha)
                if self.show_dot:
                    combined_img = self.model.draw_dots(raw_frame, rad=10)
                # combined_img = frame.copy()
                self.output_queue.put(combined_img)
            else:
                # 少し待つ
                time.sleep(0.01)

    def stop(self):
        self.stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    # parser.add_argument("--width", type=int, default=1280, help="width")
    # parser.add_argument("--height", type=int, default=720, help="height")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    parser.add_argument("--hide_background", action="store_true")
    parser.add_argument("--show_dot", action="store_true")
    parser.add_argument("--show_mask", action="store_true")

    args = parser.parse_args()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FPS, 10)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))


    # Initialize YOLOv5 Instance Segmentator
    # model_path = "./weights/yolov8s-seg.onnx"
    model_path = args.model
    yoloseg = YOLOSeg(model_path, conf_threshold=args.conf, iou_threshold=args.iou)

    # キューとスレッドの初期化
    input_queue = queue.Queue(maxsize=1)  # 入力キュー
    output_queue = queue.Queue(maxsize=1)  # 出力キュー
    inference_worker = InferenceWorker(yoloseg, input_queue, output_queue, hide_bg=args.hide_background, show_mask=args.show_mask, show_dot=args.show_dot)
    inference_worker.start()

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    #  cv2.namedWindow("Detected Objects", cv2.WINDOW_AUTOSIZE)
    
    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # キューにフレームを追加
        if input_queue.empty():
            input_queue.put(frame)

        # 結果があれば表示
        if output_queue.empty():
            resized_frame = cv2.resize(frame, (args.width, args.height))
            cv2.imshow("Detected Objects", resized_frame)
        if not output_queue.empty():
            combined_img = output_queue.get()            
            combined_img = cv2.resize(combined_img, (args.width, args.height))            
            cv2.imshow("Detected Objects", combined_img)
            cv2.imshow("Detected Objects", combined_img)
            

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # スレッドの終了
    inference_worker.stop()
    inference_worker.join()

if __name__ == "__main__":
    main()
