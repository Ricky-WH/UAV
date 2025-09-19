from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    cfg_path = '/ultralytics/cfg/models/v8/detect/yolov8.yaml'
    model = YOLO(cfg_path)
    model._new(cfg_path, task='detect', verbose=True)

