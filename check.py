import onnxruntime as ort
onnx_model="./weights/yolov8s-seg.onnx"
session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider"]
        )
print("session",    session.get_providers())