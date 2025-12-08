import json
import os
from ultralytics import YOLO
from pathlib import Path

def load_settings(json_path, model_path, dataset_path):
    with open(json_path, 'r') as f:
        settings = json.load(f)
        settings['model'] = model_path
        settings['data'] = dataset_path

    return settings

def train_yolo(
        model_path,
        dataset_yaml,
        train_json_path = './train_setting.json'
):    
    # Load settings
    settings = load_settings(train_json_path, model_path, dataset_yaml)
    
    print("=" * 60)
    print("YOLO Training Configuration")
    print("=" * 60)
    print(json.dumps(settings, indent=2))
    print("=" * 60)
    
    # Initialize model
    model_name = settings.pop('model')
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Start training
    print("\nStarting training...\n")
    model.train(**settings)
    
    # Print results
    # 自动判断任务类型来决定打印什么路径
    task = settings.get('task', 'detect') 
    # 如果没在json里指定task，yolo通常会自动识别，但路径可能变
    
    # Print results
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # 注意：Pose任务默认是在 runs/pose 下
    save_dir = model.trainer.save_dir if hasattr(model, 'trainer') else f"runs/{task}/{settings.get('name', 'train')}"
    print(f"Results saved at: {save_dir}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    
    print("\nValidation Metrics:")
    # Pose 模型主要看 pose map
    if hasattr(metrics, 'pose'):
        print(f"Pose mAP50: {metrics.pose.map50:.4f}")
        print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
    
    # 同时也可以看 Box 的指标
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)
    
    return None


def main():
    try:
        model_path = 'yolo11n-pose.pt'
        dataset_path = r"D:\LifetimerDocument\Desktop\YOLODetector\exp\yolo\data.yaml"
        train_yolo(model_path=model_path, dataset_yaml=dataset_path)

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
