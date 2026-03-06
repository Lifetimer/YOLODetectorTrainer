from tool import split_dataset, convert_labelme_to_yolo
from pathlib import Path
import os
import shutil
import zipfile

dataset_raw_path = "<your-dataset-path>"
split_ratio = 0.8

def split(source_dir, split_ratio):
    dest_base_dir = Path(__file__).parent / 'exp' / 'raw'
    
    train_dataset = dest_base_dir / 'train'
    val_dataset = dest_base_dir / 'val'
    
    split_dataset(
        source_dir=source_dir,
        split_ratio=split_ratio,
        train_dir=train_dataset,
        val_dir=val_dataset
    )

def convert():
    dest_base_dir = Path(__file__).parent / 'exp' / 'raw'
    
    train_dataset = dest_base_dir / 'train'
    val_dataset = dest_base_dir / 'val'
    output_path = dest_base_dir.parent / 'yolo'
    convert_labelme_to_yolo(
        train_path=str(train_dataset),
        val_path=str(val_dataset),
        
        output_path=str(output_path)
    )

def remove_specific_items(directory, items_to_remove: list):
    # items_to_remove = [
    #     'raw',
    #     'yolo'
    # ]

    removed_items = []
    failed_items = []

    print(f"正在清理目录: {directory}")
    print("=" * 50)

    # 移除文件夹
    for item in items_to_remove:
        item_path = os.path.join(directory, item)
        if os.path.exists(item_path):
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    removed_items.append(f"文件夹: {item}")
                    print(f"✓ 已移除文件夹: {item}")
                else:
                    os.remove(item_path)
                    removed_items.append(f"文件: {item}")
                    print(f"✓ 已移除文件: {item}")
            except Exception as e:
                failed_items.append(f"{item} - 错误: {str(e)}")
                print(f"✗ 移除 {item} 失败: {str(e)}")

    
    print("=" * 50)
    print("清理完成!")

    if removed_items:
        print(f"已成功移除 {len(removed_items)} 个项目:")
        for item in removed_items:
            print(f"  - {item}")

    if failed_items:
        print(f"\n移除失败的 {len(failed_items)} 个项目:")
        for item in failed_items:
            print(f"  - {item}")

    if not removed_items and not failed_items:
        print("没有找到需要清理的项目。")
        
def zip_directory(folder_path, output_path):
    """压缩目录到ZIP文件"""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

if __name__ == '__main__':
    if os.path.exists('YOLODataset.zip'):
        os.remove('YOLODataset.zip')
    remove_specific_items(directory='exp', items_to_remove=['raw', 'yolo'])
    split(dataset_raw_path, split_ratio)
    convert()
    remove_specific_items(directory='exp', items_to_remove=['raw'])
    shutil.copy2('tool/relocated.py', 'exp/yolo/relocated.py')
    zip_directory('exp/yolo', 'YOLODataset.zip')
