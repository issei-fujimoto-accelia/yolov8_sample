from ultralytics import YOLO
import argparse
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, required=True, help="Path to pt model")
  parser.add_argument("--train_config_path", type=str, required=True, help="Path to pt model")
  parser.add_argument("--save_dir", type=str, required=True)
  args = parser.parse_args()

  os.makedirs(args.save_dir, exist_ok=True)
  # Load a model
  model = YOLO(args.model)
  # Train the model
  results = model.train(
    data=args.train_config_path, 
    epochs=100, 
    imgsz=640,
    save_dir=args.save_dir
    )
  print(results)


if __name__ == "__main__":
  main()

