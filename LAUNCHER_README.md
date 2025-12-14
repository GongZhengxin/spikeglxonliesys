# Windows 启动脚本使用说明

本目录包含两个批处理脚本用于在 Windows 上启动 Real-time Neural Analysis System。

## 文件说明

### 1. `start_realtime_gui.bat` (推荐)
**功能强大的自动检测版本**

特点：
- ✅ 自动检测 Anaconda/Miniconda 安装位置
- ✅ 支持多种安装路径
- ✅ 详细的错误提示
- ✅ 兼容性强

适合：首次使用或不确定 conda 安装位置的用户

### 2. `start_gui_simple.bat`
**简化版本**

特点：
- ✅ 代码简洁
- ✅ 启动速度快
- ⚠️ 需要 conda 已在系统 PATH 中

适合：已配置好 conda 环境的高级用户

## 使用步骤

### 方法一：使用完整版脚本（推荐）

1. **配置环境名称**
   
   编辑 `start_realtime_gui.bat`，修改第15行：
   ```batch
   set ENV_NAME=spikeglx
   ```
   改为你的 conda 环境名称。

2. **运行脚本**
   
   双击 `start_realtime_gui.bat` 或在命令行中运行：
   ```cmd
   start_realtime_gui.bat
   ```

3. **检查输出**
   
   脚本会显示检测和启动过程：
   ```
   [1/4] Detecting Conda installation...
   [2/4] Activating conda environment...
   [3/4] Checking for Python script...
   [4/4] Environment Information...
   ```

### 方法二：使用简化版脚本

1. **确保 conda 在 PATH 中**
   
   打开命令提示符，测试：
   ```cmd
   conda --version
   ```
   应该显示版本号。

2. **配置并运行**
   
   编辑 `start_gui_simple.bat` 中的环境名称，然后双击运行。

## 常见问题解决

### ❌ 错误：找不到 conda

**症状：**
```
Could not find conda installation!
```

**解决方案：**

1. **检查 conda 是否安装**
   ```cmd
   where conda
   ```

2. **添加 conda 到 PATH**
   
   Windows 10/11:
   - 右键 "此电脑" → "属性"
   - "高级系统设置" → "环境变量"
   - 在系统变量中找到 "Path"
   - 添加以下路径（根据你的安装位置调整）：
     ```
     C:\Users\YourUsername\anaconda3
     C:\Users\YourUsername\anaconda3\Scripts
     C:\Users\YourUsername\anaconda3\Library\bin
     ```

3. **手动指定 conda 路径**
   
   在脚本开头添加：
   ```batch
   set PATH=C:\Users\YourUsername\anaconda3\Scripts;%PATH%
   ```

### ❌ 错误：无法激活环境

**症状：**
```
Failed to activate environment: spikeglx
```

**解决方案：**

1. **检查环境是否存在**
   ```cmd
   conda env list
   ```

2. **创建环境**（如果不存在）
   ```cmd
   conda create -n spikeglx python=3.10
   ```

3. **安装依赖**
   ```cmd
   conda activate spikeglx
   pip install -r requirements.txt
   ```

### ❌ 错误：找不到 Python 脚本

**症状：**
```
Could not find: RealTimeGUIv4t.py
```

**解决方案：**

1. **确保在正确目录运行**
   
   脚本必须和 `RealTimeGUIv4t.py` 在同一目录中。

2. **或修改脚本路径**
   
   在 .bat 文件中添加：
   ```batch
   cd /d "F:\#Onlinesys\online-sys\spikeglxonlinesys"
   ```

## 高级配置

### 创建桌面快捷方式

1. 右键 `start_realtime_gui.bat` → "创建快捷方式"
2. 将快捷方式拖到桌面
3. 右键快捷方式 → "属性"：
   - "起始位置" 设置为脚本所在目录
   - "运行方式" 可选择 "最小化"

### 自动切换到工作目录

在 .bat 文件开头添加：
```batch
REM 切换到脚本所在目录
cd /d "%~dp0"
```

### 使用不同的 Python 解释器

如果需要使用特定版本的 Python：
```batch
REM 使用特定 Python 版本
set PYTHON_EXE=C:\Python310\python.exe
"%PYTHON_EXE%" "%SCRIPT_NAME%"
```

## 环境变量说明

脚本中可配置的变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENV_NAME` | `spikeglx` | Conda 环境名称 |
| `SCRIPT_NAME` | `RealTimeGUIv4t.py` | Python 脚本名称 |
| `CONDA_ROOT` | 自动检测 | Conda 安装根目录 |

## 支持的 Conda 安装位置

脚本会自动检测以下位置：

1. 系统 PATH 中的 conda
2. `%USERPROFILE%\anaconda3`
3. `%USERPROFILE%\miniconda3`
4. `C:\ProgramData\anaconda3`
5. `C:\Anaconda3`
6. `C:\Miniconda3`
7. 其他驱动器 (D:, E:) 的标准位置

## 兼容性

- ✅ Windows 10
- ✅ Windows 11
- ✅ Windows Server 2016+
- ✅ Anaconda 2020.x - 2024.x
- ✅ Miniconda 4.x+

## 故障排除日志

如果遇到问题，脚本会显示详细的诊断信息。你可以：

1. 截图错误信息
2. 查看 "Environment Information" 部分
3. 运行以下命令收集信息：
   ```cmd
   conda info
   conda env list
   where python
   ```

## 创建完整的开发环境

如果需要从头开始设置：

```batch
REM 1. 创建环境
conda create -n spikeglx python=3.10 -y

REM 2. 激活环境
conda activate spikeglx

REM 3. 安装依赖
pip install PyQt5 pyqtgraph numpy scipy pandas pyyaml psutil

REM 4. 测试启动
python RealTimeGUIv4t.py
```

## 其他启动方式

### 使用 Python 直接启动
```cmd
cd F:\#Onlinesys\online-sys\spikeglxonlinesys
conda activate spikeglx
python RealTimeGUIv4t.py
```

### 使用 Anaconda Prompt
1. 打开 "Anaconda Prompt"
2. 运行：
   ```cmd
   conda activate spikeglx
   cd /d F:\#Onlinesys\online-sys\spikeglxonlinesys
   python RealTimeGUIv4t.py
   ```

## 需要帮助？

如果遇到问题：
1. 查看上面的 "常见问题解决"
2. 检查错误信息中的提示
3. 确保所有依赖都已正确安装
4. 尝试在命令行中手动运行每个步骤以定位问题

---

**提示**: 推荐使用 `start_realtime_gui.bat`，它提供最好的兼容性和错误诊断。
