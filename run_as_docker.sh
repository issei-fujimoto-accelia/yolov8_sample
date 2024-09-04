t=ultralytics/ultralytics:latest
docker run -it \
--ipc=host \
--gpus all $t \
python main.py \
--model "./models/yolov8_seg_turnip.onnx"

