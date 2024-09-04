from ultralytics import YOLO

model_path = "./weights/yolov8s-seg.pt"
model_path = "./weights/yolov8_seg_turnip.pt"

# convert to ONNX model
model = YOLO(model_path)
# model.export(format='onnx', imgsz=640, simplify=True, opset=9)
model.export(format='onnx', imgsz=[480,640], opset=20)


## vetsion対応表
## https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support