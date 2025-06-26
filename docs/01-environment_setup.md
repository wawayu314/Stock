# 步骤 1: 环境搭建和依赖安装

本节将指导您完成 StockMixer 项目的初始环境配置和所需依赖的安装。

## 1.1 Python 环境

**任务**: 确保您的系统中安装了 Python 3.7 版本。

**提示词**:
"我需要设置 Python 3.7 的开发环境。请提供在不同操作系统 (Windows, macOS, Linux) 上安装和管理 Python 3.7 版本的通用步骤。如果我已经安装了其他 Python 版本，如何创建一个独立的 Python 3.7 环境（例如使用 venv 或 conda）？"

**测试方法**:
- 在终端或命令行中执行 `python --version` 或 `python3 --version`，确认输出为 Python 3.7.x。
- 如果使用了虚拟环境，请确认虚拟环境已成功激活，并且在激活的环境中 `python --version` 显示正确的版本。

## 1.2 PyTorch 安装

**任务**: 安装 PyTorch (版本约等于 1.10.1)，并根据您的硬件情况选择合适的 CUDA 版本（如果计划使用 GPU）。

**提示词**:
"我需要为 Python 3.7 环境安装 PyTorch，版本大约为 1.10.1。请提供详细的安装指导，包括如何选择 CPU 版本或特定 CUDA 版本的 GPU 版本。请引导我访问 PyTorch 官方网站的安装向导，并解释如何根据我的系统配置选择正确的安装命令。"

**测试方法**:
- **单元测试**:
    - 准备创建一个名为 `test_pytorch_installation.py` 的文件。
    - 在该文件中，编写一个简单的 Python 脚本来导入 `torch` 模块，并打印 `torch.__version__` 和 `torch.cuda.is_available()` 的结果。
    - 运行此脚本，确认 PyTorch 版本符合要求，并且 CUDA 可用性符合您的预期（如果安装了 GPU 版本）。

## 1.3 其他 Python 依赖安装

**任务**: 根据 `requirements.txt` 文件中的列表，安装项目所需的其他 Python 包。这些包包括 `numpy` (约等于 1.21.5), `PyYAML`, `pandas`, `tqdm`, 和 `matplotlib`。

**提示词**:
"我有一个 `requirements.txt` 文件，内容如下：
tqdm==4.64.1
pandas==2.0.3
pyyaml==6.0
numpy==1.22.1
matplotlib==3.5.1
请提供使用 pip 根据此文件安装所有依赖的命令。如果某些包在安装过程中遇到特定平台的编译问题（例如在 Windows 上），请提供可能的解决方案或查找帮助的建议。"

**测试方法**:
- **单元测试**:
    - 准备创建一个名为 `test_dependencies_installation.py` 的文件。
    - 在该文件中，编写一个 Python 脚本，尝试导入 `requirements.txt` 中列出的所有库 (`tqdm`, `pandas`, `yaml`, `numpy`, `matplotlib`)。
    - 运行此脚本，确保所有库都能成功导入而没有 `ModuleNotFoundError`。
- **命令行检查**:
    - 执行 `pip freeze` 命令，并检查输出中是否包含 `requirements.txt` 中列出的所有包及其指定的版本（或兼容版本）。

完成以上步骤后，您的开发环境应该已经准备就绪，可以开始项目的后续开发工作。 