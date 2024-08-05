# readme

## train
### dataset
labelmeでアノテーションしたディレクトリに対してlabelme2yoloで変換

`labelme2yolo --json_dir ./resize_images/ --val_size 0.15 --test_size 0.15 --output_format bbox`

https://github.com/GreatV/labelme2yolo/tree/main
