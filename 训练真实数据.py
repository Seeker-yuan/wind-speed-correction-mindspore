#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用真实wind_data数据训练和验证时空图神经网络
"""

import os
import numpy as np
import pandas as pd
from mindspore_gnn_model import MindSporeGNNPredictor

def load_wind_data():
    """加载所有104台风机的真实数据"""
    wind_data_path = r'C:\Users\31876\Desktop\风能ui设计\wind_data'
    
    print("=" * 70)
    print("加载真实风机数据...")
    print("=" * 70)
    
    machines = []
    positions = []
    
    # 读取所有风机数据文件
    for i in range(1, 105):
        file_name = f"{i}号机组.xlsx"
        file_path = os.path.join(wind_data_path, file_name)
        
        try:
            df = pd.read_excel(file_path)
            
            # 提取风速数据（假设在'风速'或类似列中）
            if '风速' in df.columns:
                wind_speed = df['风速'].values
            elif '测风仪风速' in df.columns:
                wind_speed = df['测风仪风速'].values
            else:
                # 使用第一个数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                wind_speed = df[numeric_cols[0]].values
            
            # 提取位置信息（如果有）
            if '经度' in df.columns and '纬度' in df.columns:
                lon = df['经度'].iloc[0] if not pd.isna(df['经度'].iloc[0]) else i * 0.01
                lat = df['纬度'].iloc[0] if not pd.isna(df['纬度'].iloc[0]) else i * 0.01
            else:
                # 使用模拟位置
                lon = (i % 13) * 0.01
                lat = (i // 13) * 0.01
            
            machines.append(wind_speed)
            positions.append([lon, lat])
            
            print(f"✓ 加载 {file_name}: {len(wind_speed)} 个数据点")
            
        except Exception as e:
            print(f"✗ 加载 {file_name} 失败: {e}")
    
    print(f"\n成功加载 {len(machines)} 台风机数据")
    print("=" * 70)
    
    return machines, np.array(positions)


def prepare_temporal_data(machines, seq_len=24):
    """
    准备时序数据
    
    Args:
        machines: 所有风机的数据 [n_machines, n_timepoints]
        seq_len: 时序窗口长度
    
    Returns:
        X: [n_samples, n_machines, seq_len, 1] 输入特征
        y: [n_samples] 目标值
    """
    n_machines = len(machines)
    min_len = min(len(m) for m in machines)
    
    # 对齐所有风机数据长度
    aligned_data = np.zeros((n_machines, min_len))
    for i, machine_data in enumerate(machines):
        aligned_data[i, :] = machine_data[:min_len]
    
    # 归一化
    mean = np.nanmean(aligned_data)
    std = np.nanstd(aligned_data)
    aligned_data = (aligned_data - mean) / (std + 1e-8)
    
    # 构建时序样本
    X_list = []
    y_list = []
    target_machine_idx = []
    
    # 为每台风机创建训练样本
    for target_idx in range(n_machines):
        for t in range(seq_len, min_len):
            # 输入：所有风机的历史seq_len时间步数据
            X_sample = aligned_data[:, t-seq_len:t]  # [n_machines, seq_len]
            X_sample = X_sample[:, :, np.newaxis]  # [n_machines, seq_len, 1]
            
            # 输出：目标风机的当前时刻数据
            y_sample = aligned_data[target_idx, t]
            
            X_list.append(X_sample)
            y_list.append(y_sample)
            target_machine_idx.append(target_idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    target_machine_idx = np.array(target_machine_idx)
    
    return X, y, target_machine_idx, (mean, std)


def train_and_evaluate():
    """训练和评估模型"""
    
    # 1. 加载真实数据
    machines, positions = load_wind_data()
    
    # 2. 准备时序数据
    print("\n准备时序训练数据...")
    X, y, target_idx, norm_params = prepare_temporal_data(machines, seq_len=24)
    
    n_samples = X.shape[0]
    print(f"  总样本数: {n_samples}")
    print(f"  风机数量: {X.shape[1]}")
    print(f"  时序长度: {X.shape[2]}")
    print(f"  归一化参数: mean={norm_params[0]:.2f}, std={norm_params[1]:.2f}")
    
    # 3. 划分训练集和测试集（80/20）
    split_idx = int(0.8 * n_samples)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    target_train = target_idx[:split_idx]
    
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    target_test = target_idx[split_idx:]
    
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 4. 创建并训练模型
    print("\n" + "=" * 70)
    print("训练时空图神经网络（ST-GNN）...")
    print("=" * 70)
    
    model = MindSporeGNNPredictor(
        n_features=1,
        hidden_dim=128,
        n_layers=4,
        seq_len=24
    )
    
    # 训练（使用前100个样本快速验证，你可以改成全部数据）
    train_size = min(500, len(X_train))
    model.fit(
        X_train[:train_size],
        y_train[:train_size],
        positions,
        target_train[:train_size],
        epochs=50,
        learning_rate=0.001
    )
    
    # 5. 测试评估
    print("\n" + "=" * 70)
    print("在测试集上评估模型...")
    print("=" * 70)
    
    test_size = min(100, len(X_test))
    predictions = []
    
    for i in range(test_size):
        pred = model.predict(X_test[i], positions, target_test[i])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    true_values = y_test[:test_size]
    
    # 计算误差（归一化空间）
    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))
    
    # 反归一化到实际风速
    mean, std = norm_params
    predictions_real = predictions * std + mean
    true_values_real = true_values * std + mean
    
    mse_real = np.mean((predictions_real - true_values_real) ** 2)
    rmse_real = np.sqrt(mse_real)
    mae_real = np.mean(np.abs(predictions_real - true_values_real))
    
    print(f"\n归一化空间性能:")
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    
    print(f"\n实际风速性能:")
    print(f"  MSE  = {mse_real:.4f} (m/s)²")
    print(f"  RMSE = {rmse_real:.4f} m/s")
    print(f"  MAE  = {mae_real:.4f} m/s")
    
    # 计算相对误差
    mean_true = np.mean(np.abs(true_values_real))
    relative_error = (mae_real / mean_true) * 100
    print(f"  平均相对误差 = {relative_error:.2f}%")
    
    # 显示几个预测示例
    print(f"\n{'='*70}")
    print("预测示例（前10个）:")
    print(f"{'='*70}")
    print(f"{'序号':<6} {'真实值':>10} {'预测值':>10} {'误差':>10} {'风机编号':>10}")
    print("-" * 70)
    
    for i in range(min(10, test_size)):
        error = predictions_real[i] - true_values_real[i]
        print(f"{i+1:<6} {true_values_real[i]:>10.2f} {predictions_real[i]:>10.2f} {error:>10.2f} {target_test[i]+1:>10}")
    
    print("=" * 70)
    
    # 6. 保存模型
    model_path = r'C:\Users\31876\Desktop\风能ui设计\trained_stgnn_model.ckpt'
    # 这里可以添加模型保存逻辑
    
    print(f"\n✓ 训练完成！")
    print(f"  模型架构: 2层LSTM + 4层图卷积 + 注意力机制 + 残差连接")
    print(f"  训练样本: {train_size}")
    print(f"  测试样本: {test_size}")
    print(f"  预测精度: {relative_error:.2f}% 相对误差")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
