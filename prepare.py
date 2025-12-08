from tool import split_dataset, convert_labelme_to_yolo
from pathlib import Path


def split():
    source_dir = r"D:\LifetimerDocument\Desktop\Presentation\DatasetAll"
    
    dest_base_dir = Path(__file__).parent / 'exp' / 'raw'
    
    train_dataset = dest_base_dir / 'train'
    val_dataset = dest_base_dir / 'val'
    
    split_dataset(
        source_dir=source_dir,
        split_ratio=0.8,
        train_dir=train_dataset,
        val_dir=val_dataset
    )

def convert():
    convert_labelme_to_yolo(
        train_path=r"D:\LifetimerDocument\Desktop\YOLODetector\exp\raw\train",
        val_path=r"D:\LifetimerDocument\Desktop\YOLODetector\exp\raw\val",
        
        output_path=r"D:\LifetimerDocument\Desktop\YOLODetector\exp\yolo"
    )

if __name__ == '__main__':
    convert()
