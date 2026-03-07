# 风能发电机风速误差校正系统

基于华为 MindSpore 深度学习框架的风速数据误差校正与补全系统

## 📋 项目简介

本项目使用华为 MindSpore 框架实现风力发电机测量风速的误差校正，通过深度神经网络和多特征融合技术，对缺失的风速数据进行智能补全。

## 🚀 核心技术

- **深度学习框架**: 华为 MindSpore 2.8.0
- **模型架构**: 4层全连接神经网络 + BatchNorm + Dropout
- **特征融合**: DTW时序相似度 + Pearson相关系数 + 地理位置距离
- **优化策略**: Adam优化器 + 集成学习

## 📊 性能指标

- **预测误差率**: 1-2%
- **MSE**: 0.057
- **训练收敛**: Loss从0.17降至0.10

## 🔧 环境要求

- Python 3.9
- MindSpore 2.8.0
- pandas, numpy, scikit-learn, openpyxl, fastdtw

## 📦 安装步骤

### 1. 安装 Anaconda

下载地址: https://www.anaconda.com/download

### 2. 创建环境并安装依赖

```bash
# 创建Python 3.9环境
conda create -n mindspore python=3.9 -y

# 激活环境
conda activate mindspore

# 安装MindSpore
pip install mindspore==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
pip install pandas numpy scikit-learn openpyxl fastdtw -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🎯 使用方法

### 方式1: 双击运行（推荐）

直接双击 `运行项目.bat` 文件

### 方式2: 命令行运行

```bash
# 激活环境
conda activate mindspore

# 测试模型
python mindspore_model_v2.py

# 运行完整项目
python 汇总预测_mindspore版.py
```

## 📁 项目结构

```
.
├── mindspore_model_v2.py          # MindSpore神经网络模型
├── 汇总预测_mindspore版.py         # 主程序（数据处理+补全）
├── wind_data/                     # 原始数据目录
└── cleaned_data/                  # 输出数据目录
```

## 🎓 模型架构

```
输入层 (4个邻近风机)
    ↓
全连接层 (64) + BatchNorm + ReLU + Dropout
    ↓
全连接层 (64) + BatchNorm + ReLU + Dropout
    ↓
全连接层 (32) + BatchNorm + ReLU
    ↓
输出层 (1) - 预测风速
```

## 📈 工作流程

1. 预加载所有风机数据到内存
2. 识别每台风机的缺失数据块
3. 基于DTW、Pearson、地理位置选择候选风机
4. 使用MindSpore神经网络训练预测模型
5. 三路预测结果融合
6. 补全缺失数据并保存
7. 生成缺损率分析报告

## 🔬 技术特点

### 创新点

1. **华为MindSpore框架** - 支持国产化，可对接华为云生态
2. **多特征融合** - 时序+统计+空间三维度特征
3. **集成学习** - 三路预测加权平均，提升鲁棒性
4. **自适应训练** - 根据数据量动态调整批次大小

### 优势

- ✅ 预测精度高（误差1-2%）
- ✅ 训练稳定（Loss收敛良好）
- ✅ 可扩展性强（易于升级到LSTM、GNN）
- ✅ 国产化支持（MindSpore生态）
