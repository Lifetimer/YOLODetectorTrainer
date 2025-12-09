# YOLODetectorTrainer

YOLODetectorTrainer, 在ultralytics的YOLO的基础上添加了数据集预处理、默认训练参数和常见配置，旨在使初学者使用更为方便。

## 数据集准备

需要一个只包含图片的文件夹，文件夹内是未标注的图片。

## LabelMe标注

### 1. 新建venv虚拟环境

```Shell
python -m venv venv
```

### 2. 激活虚拟环境

Windows:
```PowerShell
./venv/Scripts/activate
```

Linux:
```Bash
./venv/bin/activate
```

### 2. 安装labelme

```Shell
pip3 install labelme
```

### 3. 启动LabelMe

```
labelme
```

## 数据集预处理

修改prepare.py文件，依次修改主函数调用split和convert这两个函数。需要修改路径信息。

## 训练指南

### 1. 先创建虚拟环境
```PowerShell
python -m venv venv
```
### 2. 激活虚拟环境

Windows:
```PowerShell
./venv/Scripts/activate
```

Linux:
```Bash
./venv/bin/activate
```
### 3. 安装torch
```PowerShell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
### 4. 安装其它依赖
```PowerShell
pip3 install -r requirements.txt
```