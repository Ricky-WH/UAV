from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    cfg_path = '/Users/ricky-hang/study/Anti-UAV/Anti-UAV/ultralytics/cfg/models/11/yolo11-BLIP-v1.yaml'
    model = YOLO(cfg_path)
    model._new(cfg_path, task='detect', verbose=True)

