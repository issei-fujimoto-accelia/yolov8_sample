from ultralytics import YOLO
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, required=True, help="Path to pt model")
  parser.add_argument("--train_config_path", type=str, required=True, help="Path to pt model")
  args = parser.parse_args()

  # Load a model
  model = YOLO(args.model)
  # Train the model
  results = model.train(data=args.train_config_path, epochs=100, imgsz=640)
  print(results)


if __name__ == "__main__":
  main()

