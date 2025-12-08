## 部署指南

1. 先创建虚拟环境
```PowerShell
python -m venv venv
```
2. 激活虚拟环境
Windows:
```PowerShell
./venv/Scripts/activate
```

Linux:
```PowerShell
./venv/bin/activate
```
3. 安装torch
```PowerShell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
4. 安装其它依赖
```PowerShell
pip3 install -r requirements.txt
```