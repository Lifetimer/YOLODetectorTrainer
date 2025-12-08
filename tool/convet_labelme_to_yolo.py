import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


def collect_labels_and_keypoints(train_path: str, val_path: str) -> Tuple[Set[str], Set[str], bool]:
    """
    扫描数据集，收集所有物体类别和关键点类别
    
    Args:
        train_path: 训练集路径
        val_path: 验证集路径
    
    Returns:
        object_classes: 物体类别集合
        keypoint_classes: 关键点类别集合
        has_keypoints: 是否包含关键点
    """
    object_classes = set()
    keypoint_classes = set()
    
    for dataset_path in [train_path, val_path]:
        if not os.path.exists(dataset_path):
            continue
        
        for file in os.listdir(dataset_path):
            if not file.endswith('.json'):
                continue
            
            json_path = os.path.join(dataset_path, file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for shape in data.get('shapes', []):
                    label = shape['label']
                    shape_type = shape['shape_type']
                    
                    if shape_type == 'rectangle':
                        object_classes.add(label)
                    elif shape_type == 'point':
                        keypoint_classes.add(label)
            except Exception as e:
                print(f"警告: 读取文件失败 {json_path}: {e}")
                continue
    
    has_keypoints = len(keypoint_classes) > 0
    return object_classes, keypoint_classes, has_keypoints


def build_keypoint_mapping(object_classes: Set[str], keypoint_classes: Set[str]) -> Tuple[Dict[str, Tuple[str, int]], Dict[str, int]]:
    """
    询问用户建立关键点与物体的映射关系
    
    Args:
        object_classes: 物体类别集合
        keypoint_classes: 关键点类别集合
    
    Returns:
        mapping: {keypoint_label: (object_class, keypoint_index)}
        keypoint_counts: {object_class: keypoint_count}
    """
    mapping = {}
    object_classes_list = sorted(list(object_classes))
    
    # 为每个物体类别维护关键点计数器
    keypoint_counters = {obj_class: 0 for obj_class in object_classes_list}
    
    print("\n" + "=" * 60)
    print("关键点与物体类别映射配置")
    print("=" * 60)
    print(f"\n检测到的物体类别: {', '.join(object_classes_list)}")
    print(f"检测到的关键点类别: {', '.join(sorted(keypoint_classes))}\n")
    
    for kpt_label in sorted(keypoint_classes):
        while True:
            print(f"\n关键点 '{kpt_label}' 属于哪个物体类别？")
            for idx, obj_class in enumerate(object_classes_list):
                print(f"  {idx}: {obj_class}")
            
            try:
                choice = input("请输入编号: ").strip()
                choice_idx = int(choice)
                
                if 0 <= choice_idx < len(object_classes_list):
                    obj_class = object_classes_list[choice_idx]
                    kpt_idx = keypoint_counters[obj_class]
                    keypoint_counters[obj_class] += 1
                    
                    mapping[kpt_label] = (obj_class, kpt_idx)
                    print(f"  ✓ '{kpt_label}' -> '{obj_class}' (索引: {kpt_idx})")
                    break
                else:
                    print(f"  ✗ 请输入 0-{len(object_classes_list) - 1} 之间的数字")
            except ValueError:
                print("  ✗ 请输入有效的数字")
            except (KeyboardInterrupt, EOFError):
                print("\n\n用户中断操作")
                exit(1)
    
    return mapping, keypoint_counters


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """计算两点间的欧氏距离"""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_box_center(box_points: List[List[float]]) -> List[float]:
    """获取矩形框的中心点坐标"""
    x1, y1 = box_points[0]
    x2, y2 = box_points[1]
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def assign_keypoints_to_boxes(rectangles: List[dict], points: List[dict],
                               keypoint_mapping: Dict[str, Tuple[str, int]]) -> List[dict]:
    """
    将关键点分配给对应的检测框
    
    Args:
        rectangles: 所有矩形框
        points: 所有关键点
        keypoint_mapping: 关键点到物体类别的映射
    
    Returns:
        annotations: 标注列表，每项包含 {class, bbox, keypoints}
    """
    # 按物体类别分组rectangles
    boxes_by_class = {}
    for rect in rectangles:
        obj_class = rect['label']
        if obj_class not in boxes_by_class:
            boxes_by_class[obj_class] = []
        boxes_by_class[obj_class].append(rect)
    
    # 按物体类别分组points
    points_by_class = {}
    for point in points:
        kpt_label = point['label']
        if kpt_label in keypoint_mapping:
            obj_class, kpt_idx = keypoint_mapping[kpt_label]
            if obj_class not in points_by_class:
                points_by_class[obj_class] = []
            points_by_class[obj_class].append({
                'coords': point['points'][0],
                'label': kpt_label,
                'kpt_idx': kpt_idx
            })
    
    # 为每个框分配关键点
    annotations = []
    
    for obj_class, boxes in boxes_by_class.items():
        obj_points = points_by_class.get(obj_class, [])
        
        # 只有一个框时，所有关键点都分配给它
        if len(boxes) == 1:
            box = boxes[0]
            annotations.append({
                'class': obj_class,
                'bbox': box['points'],
                'keypoints': obj_points
            })
        else:
            # 多个框时，按距离最近原则分配关键点
            # 初始化每个框的标注
            box_annotations = []
            for box in boxes:
                box_annotations.append({
                    'class': obj_class,
                    'bbox': box['points'],
                    'keypoints': [],
                    'center': get_box_center(box['points'])
                })
            
            # 将每个关键点分配给最近的框
            for point_info in obj_points:
                point_coords = point_info['coords']
                min_distance = float('inf')
                closest_idx = 0
                
                for idx, box_ann in enumerate(box_annotations):
                    distance = calculate_distance(point_coords, box_ann['center'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
                
                box_annotations[closest_idx]['keypoints'].append(point_info)
            
            # 移除临时的center字段
            for box_ann in box_annotations:
                del box_ann['center']
                annotations.append(box_ann)
    
    return annotations


def convert_to_yolo_format(annotation: dict, img_width: int, img_height: int,
                           class_to_idx: Dict[str, int], num_keypoints: int) -> str:
    """
    将标注转换为YOLO格式字符串
    
    Args:
        annotation: 单个物体的标注信息
        img_width: 图像宽度
        img_height: 图像高度
        class_to_idx: 类别名到索引的映射
        num_keypoints: 该物体类别的关键点数量
    
    Returns:
        YOLO格式的标注行
    """
    obj_class = annotation['class']
    class_idx = class_to_idx[obj_class]
    
    # 归一化bbox坐标
    bbox = annotation['bbox']
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    # 确保坐标顺序正确
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # 转换为YOLO格式：中心点坐标和宽高（归一化）
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    line = f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    # 如果有关键点
    if num_keypoints > 0:
        # 初始化关键点数组，默认为未标注(0, 0, 0)
        keypoints_array = [[0.0, 0.0, 0] for _ in range(num_keypoints)]
        
        # 填充实际标注的关键点
        for kpt_info in annotation['keypoints']:
            kpt_idx = kpt_info['kpt_idx']
            x, y = kpt_info['coords']
            
            # 归一化坐标
            x_norm = x / img_width
            y_norm = y / img_height
            
            # 可见性标志为2（可见）
            keypoints_array[kpt_idx] = [x_norm, y_norm, 2]
        
        # 追加关键点到输出行
        for kpt in keypoints_array:
            line += f" {kpt[0]:.6f} {kpt[1]:.6f} {kpt[2]}"
    
    return line


def process_dataset(train_path: str, val_path: str, output_path: str,
                    keypoint_mapping: Optional[Dict] = None,
                    keypoint_counts: Optional[Dict] = None,
                    class_to_idx: Optional[Dict] = None,
                    has_keypoints: bool = False):
    """
    处理数据集，转换为YOLO格式
    
    Args:
        train_path: 训练集路径
        val_path: 验证集路径
        output_path: 输出路径
        keypoint_mapping: 关键点映射
        keypoint_counts: 每个类别的关键点数量
        class_to_idx: 类别到索引的映射
        has_keypoints: 是否包含关键点
    """
    output_path = Path(output_path)
    
    # 创建目录结构
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    for split, dataset_path in [('train', train_path), ('val', val_path)]:
        if not os.path.exists(dataset_path):
            print(f"警告: {split} 路径不存在: {dataset_path}")
            continue
        
        print(f"\n处理 {split} 数据集...")
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(dataset_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        processed_count = 0
        for img_file in image_files:
            img_name = Path(img_file).stem
            img_path = os.path.join(dataset_path, img_file)
            json_path = os.path.join(dataset_path, f"{img_name}.json")
            
            # 复制图片
            dst_img_path = output_path / 'images' / split / img_file
            shutil.copy2(img_path, dst_img_path)
            
            # 处理标注
            txt_path = output_path / 'labels' / split / f"{img_name}.txt"
            
            if not os.path.exists(json_path):
                # 负样本，创建空文件
                txt_path.write_text('', encoding='utf-8')
                processed_count += 1
                continue
            
            # 读取json标注
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  警告: 读取JSON失败 {json_path}: {e}")
                txt_path.write_text('', encoding='utf-8')
                continue
            
            img_width = data.get('imageWidth', 0)
            img_height = data.get('imageHeight', 0)
            
            if img_width == 0 or img_height == 0:
                print(f"  警告: 图像尺寸无效 {json_path}")
                txt_path.write_text('', encoding='utf-8')
                continue
            
            # 分离rectangle和point
            rectangles = []
            points = []
            
            for shape in data.get('shapes', []):
                if shape['shape_type'] == 'rectangle':
                    rectangles.append(shape)
                elif shape['shape_type'] == 'point':
                    points.append(shape)
            
            # 如果没有检测框，创建空文件
            if not rectangles:
                txt_path.write_text('', encoding='utf-8')
                processed_count += 1
                continue
            
            # 生成标注
            annotations = []
            
            if has_keypoints and keypoint_mapping:
                # 关键点检测模式
                annotations = assign_keypoints_to_boxes(rectangles, points, keypoint_mapping)
            else:
                # 普通目标检测模式
                for rect in rectangles:
                    annotations.append({
                        'class': rect['label'],
                        'bbox': rect['points'],
                        'keypoints': []
                    })
            
            # 写入txt文件
            with open(txt_path, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    obj_class = ann['class']
                    num_kpts = keypoint_counts.get(obj_class, 0) if keypoint_counts else 0
                    
                    line = convert_to_yolo_format(ann, img_width, img_height,
                                                  class_to_idx, num_kpts)
                    f.write(line + '\n')
            
            processed_count += 1
        
        print(f"  完成 {processed_count}/{len(image_files)} 张图片")


def generate_data_yaml(output_path: str, class_to_idx: Dict[str, int],
                       keypoint_counts: Optional[Dict[str, int]] = None):
    """
    生成YOLO数据集配置文件data.yaml
    
    Args:
        output_path: 输出路径
        class_to_idx: 类别到索引的映射
        keypoint_counts: 每个类别的关键点数量（可选）
    """
    output_path = Path(output_path)
    
    yaml_content = f"path: {output_path.absolute()}\n"
    yaml_content += "train: images/train\n"
    yaml_content += "val: images/val\n"
    yaml_content += "\n"
    
    # 如果是关键点检测
    if keypoint_counts and any(count > 0 for count in keypoint_counts.values()):
        # 使用所有类别中最大的关键点数量
        max_keypoints = max(keypoint_counts.values())
        yaml_content += f"kpt_shape: [{max_keypoints}, 3]\n"
        
        # 生成flip_idx（简单递增序列）
        flip_idx = list(range(max_keypoints))
        yaml_content += f"flip_idx: {flip_idx}\n"
        yaml_content += "\n"
    
    # 类别名称
    yaml_content += "names:\n"
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        yaml_content += f"  {idx}: {class_name}\n"
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n生成配置文件: {yaml_path}")


def convert_labelme_to_yolo(train_path: str, val_path: str, output_path: str):
    """
    将LabelMe格式的数据集转换为YOLO格式
    
    Args:
        train_path: 训练集目录路径
        val_path: 验证集目录路径
        output_path: 输出父目录路径
    """
    print("=" * 60)
    print("LabelMe to YOLO 数据集转换工具")
    print("=" * 60)
    
    # 阶段1：扫描数据集
    print("\n阶段 1/3: 扫描数据集...")
    object_classes, keypoint_classes, has_keypoints = collect_labels_and_keypoints(
        train_path, val_path
    )
    
    if not object_classes:
        print("\n错误: 未检测到任何物体类别（rectangle标注）")
        return
    
    print(f"\n检测到 {len(object_classes)} 个物体类别: {', '.join(sorted(object_classes))}")
    print(f"检测到 {len(keypoint_classes)} 个关键点类别: {', '.join(sorted(keypoint_classes))}")
    print(f"数据集模式: {'关键点检测' if has_keypoints else '目标检测'}")
    
    # 建立类别索引（按字母顺序）
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(object_classes))}
    
    # 阶段2：配置关键点映射
    keypoint_mapping = None
    keypoint_counts = {obj_class: 0 for obj_class in object_classes}
    
    if has_keypoints:
        print("\n阶段 2/3: 配置关键点映射...")
        keypoint_mapping, keypoint_counts = build_keypoint_mapping(
            object_classes, keypoint_classes
        )
        
        print("\n关键点配置总结:")
        for obj_class in sorted(object_classes):
            print(f"  {obj_class}: {keypoint_counts[obj_class]} 个关键点")
    else:
        print("\n阶段 2/3: 跳过关键点配置（数据集无关键点标注）")
    
    # 阶段3：转换数据集
    print("\n阶段 3/3: 转换数据集到 YOLO 格式...")
    process_dataset(train_path, val_path, output_path,
                   keypoint_mapping, keypoint_counts, class_to_idx, has_keypoints)
    
    # 生成配置文件
    generate_data_yaml(output_path, class_to_idx, keypoint_counts if has_keypoints else None)
    
    print("\n" + "=" * 60)
    print("✓ 转换完成！")
    print("=" * 60)
    print("\n输出目录: {os.path.abspath(output_path)}")
    print("  - images/train/  : 训练集图片")
    print("  - images/val/    : 验证集图片")
    print("  - labels/train/  : 训练集标签")
    print("  - labels/val/    : 验证集标签")
    print("  - data.yaml      : 数据集配置文件")
    print()


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description='将LabelMe格式的数据集转换为YOLO格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python %(prog)s --train /path/to/train --val /path/to/val --output /path/to/output

注意事项:
  - 支持普通目标检测和关键点检测
  - 如果JSON中没有关键点标注，将执行普通目标检测转换
  - 没有JSON文件的图片将被视为负样本（生成空标签文件）
  - 所有关键点的可见性默认为2（可见）
        """
    )
    
    parser.add_argument('--train', required=True, help='训练集目录路径')
    parser.add_argument('--val', required=True, help='验证集目录路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.train):
        print(f"错误: 训练集路径不存在: {args.train}")
        return
    
    if not os.path.exists(args.val):
        print(f"错误: 验证集路径不存在: {args.val}")
        return
    
    convert_labelme_to_yolo(args.train, args.val, args.output)


if __name__ == '__main__':
    main()
