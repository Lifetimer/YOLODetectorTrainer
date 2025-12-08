import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(source_dir, split_ratio, train_dir, val_dir, seed = 42):
    """
    分割数据集
    
    Args:
        source_dir: 源目录，包含图片和json文件
        split_ratio: 训练集比例 (0-1之间的浮点数)
        train_dir: 训练集目标目录
        val_dir: 验证集目标目录
    """
    if not 0 < split_ratio < 1:
        print("错误: split_ratio 必须在 0 和 1 之间")
        return
    
    random.seed(seed)
    
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    image_files = []
    for file in source_path.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    train_without_json = 0
    val_without_json = 0
    
    print("\n正在复制训练集文件...")
    for img_file in train_files:
        json_file = source_path / f"{img_file.stem}.json"
        shutil.copy2(img_file, train_path / img_file.name)
        if json_file.exists():
            shutil.copy2(json_file, train_path / json_file.name)
        else:
            train_without_json += 1
    
    print("正在复制验证集文件...")
    for img_file in val_files:
        json_file = source_path / f"{img_file.stem}.json"
        shutil.copy2(img_file, val_path / img_file.name)
        if json_file.exists():
            shutil.copy2(json_file, val_path / json_file.name)
        else:
            val_without_json += 1
    
    print("\n" + "="*50)
    print("数据集分割完成!")
    print("="*50)
    print(f"训练集样本个数: {len(train_files)}")
    print(f"验证集样本个数: {len(val_files)}")
    print(f"训练集中没有对应JSON文件的图片数: {train_without_json}")
    print(f"验证集中没有对应JSON文件的图片数: {val_without_json}")
    print(f"训练集比例: {split_ratio:.2f} ({len(train_files)}/{len(image_files)})")
    print(f"验证集比例: {(1-split_ratio):.2f} ({len(val_files)}/{len(image_files)})")


def main():
    parser = argparse.ArgumentParser(description='分割数据集为训练集和验证集')
    parser.add_argument('--source_dir', type=str, required=True, 
                       help='源目录，包含图片和JSON文件')
    parser.add_argument('--split_ratio', type=float, required=True,
                       help='训练集比例 (0-1之间的浮点数，如0.8)')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='训练集目标目录')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='验证集目标目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    args = parser.parse_args()
    
    split_dataset(args.source_dir, args.split_ratio, args.train_dir, args.val_dir, args.seed)

if __name__ == "__main__":
    main()
