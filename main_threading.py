import cv2
import threading
import queue

# from models.yolo import YOLOv8Seg
from models.yoloSeg import YOLOSeg
import argparse

class InferenceWorker(threading.Thread):
    def __init__(self, model, input_queue, output_queue):
        threading.Thread.__init__(self)
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                boxes, scores, class_ids, masks = self.model(frame)
                combined_img = self.model.draw_masks_only(frame.copy())  # modelにdraw_masks_only関数がある前提
                self.output_queue.put(combined_img)
            else:
                # 少し待つ
                time.sleep(0.01)

    def stop(self):
        self.stop_event.set()

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

    # キューとスレッドの初期化
    input_queue = queue.Queue(maxsize=1)  # 入力キュー
    output_queue = queue.Queue(maxsize=1)  # 出力キュー
    inference_worker = InferenceWorker(yoloseg, input_queue, output_queue)
    inference_worker.start()

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # キューにフレームを追加
        if input_queue.empty():
            input_queue.put(frame)

        # 結果があれば表示
        if not output_queue.empty():
            combined_img = output_queue.get()
            cv2.imshow("Detected Objects", combined_img)

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # スレッドの終了
    inference_worker.stop()
    inference_worker.join()

if __name__ == "__main__":
    main()
