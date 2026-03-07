# mindspore_model_v2.py
"""
MindSpore风速预测模型 - 优化版
专为风能发电机测量风速误差校正设计
"""

import numpy as np

# 尝试导入MindSpore
try:
    import mindspore
    from mindspore import nn, ops, Tensor, context
    from mindspore import dtype as mstype
    from mindspore.train import Model
    from mindspore.nn import MSELoss, Adam
    
    # 设置运行模式
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    MINDSPORE_AVAILABLE = True
    print("✓ MindSpore已加载 (CPU模式)")
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("⚠ MindSpore未安装，使用sklearn作为后备")
    from sklearn.neural_network import MLPRegressor


class MindSporeWindPredictor:
    """
    基于MindSpore的风速预测模型
    """
    
    def __init__(self, n_neighbors=4, hidden_size=64):
        """
        初始化模型
        
        Args:
            n_neighbors: 邻近风机数量
            hidden_size: 隐藏层维度
        """
        self.n_neighbors = n_neighbors
        self.hidden_size = hidden_size
        
        if not MINDSPORE_AVAILABLE:
            # 使用sklearn作为后备
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden_size, hidden_size, 32),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
            self.framework = 'sklearn'
            print("  使用框架: sklearn (后备)")
        else:
            # 使用MindSpore
            self._build_mindspore_model()
            self.framework = 'mindspore'
            print("  使用框架: MindSpore")
        
        # 标准化参数
        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None
    
    def _build_mindspore_model(self):
        """构建MindSpore神经网络"""
        
        class WindNet(nn.Cell):
            """风速预测网络"""
            
            def __init__(self, n_in, hidden):
                super(WindNet, self).__init__()
                
                # 网络层
                self.fc1 = nn.Dense(n_in, hidden, weight_init='xavier_uniform')
                self.fc2 = nn.Dense(hidden, hidden, weight_init='xavier_uniform')
                self.fc3 = nn.Dense(hidden, 32, weight_init='xavier_uniform')
                self.fc4 = nn.Dense(32, 1, weight_init='xavier_uniform')
                
                # 激活函数
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=0.2)
                
                # Batch Normalization
                self.bn1 = nn.BatchNorm1d(hidden)
                self.bn2 = nn.BatchNorm1d(hidden)
                self.bn3 = nn.BatchNorm1d(32)
            
            def construct(self, x):
                """前向传播"""
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.dropout(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = self.relu(x)
                x = self.dropout(x)
                
                x = self.fc3(x)
                x = self.bn3(x)
                x = self.relu(x)
                
                x = self.fc4(x)
                return x
        
        # 创建网络
        self.net = WindNet(self.n_neighbors, self.hidden_size)
        
        # 损失函数和优化器
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.net.trainable_params(), learning_rate=0.001)
    
    def _normalize(self, X, y=None, fit=False):
        """数据标准化"""
        X = np.array(X, dtype=np.float32)
        
        if fit:
            self.mean_X = X.mean(axis=0, keepdims=True)
            self.std_X = X.std(axis=0, keepdims=True) + 1e-8
        
        X_norm = (X - self.mean_X) / self.std_X
        
        if y is not None:
            y = np.array(y, dtype=np.float32).reshape(-1, 1)
            if fit:
                self.mean_y = y.mean()
                self.std_y = y.std() + 1e-8
            y_norm = (y - self.mean_y) / self.std_y
            return X_norm, y_norm
        
        return X_norm
    
    def _denormalize_y(self, y_norm):
        """反标准化目标值"""
        return y_norm * self.std_y + self.mean_y
    
    def fit(self, X, y, epochs=50, batch_size=32, verbose=True):
        """
        训练模型
        
        Args:
            X: 输入特征 [n_samples, n_neighbors]
            y: 目标值 [n_samples]
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程
        """
        # 数据标准化
        X_norm, y_norm = self._normalize(X, y, fit=True)
        
        if self.framework == 'sklearn':
            # sklearn训练
            self.model.fit(X_norm, y_norm.ravel())
            if verbose:
                print("  ✓ 训练完成")
            return
        
        # MindSpore训练
        n_samples = X_norm.shape[0]
        self.net.set_train(True)
        
        # 定义前向函数
        def forward_fn(data, label):
            pred = self.net(data)
            loss = self.loss_fn(pred, label)
            return loss, pred
        
        # 获取梯度函数
        grad_fn = mindspore.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        
        # 训练循环
        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_idx = indices[i:end_idx]
                
                # 准备批次数据
                batch_X = Tensor(X_norm[batch_idx], mstype.float32)
                batch_y = Tensor(y_norm[batch_idx], mstype.float32)
                
                # 前向传播和反向传播
                (loss, _), grads = grad_fn(batch_X, batch_y)
                
                # 更新参数
                self.optimizer(grads)
                
                epoch_loss += loss.asnumpy()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if verbose:
            print("  ✓ 训练完成")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入特征 [n_samples, n_neighbors]
            
        Returns:
            predictions: 预测值 [n_samples]
        """
        # 标准化输入
        X_norm = self._normalize(X, fit=False)
        
        if self.framework == 'sklearn':
            # sklearn预测
            y_pred_norm = self.model.predict(X_norm).reshape(-1, 1)
        else:
            # MindSpore预测
            self.net.set_train(False)
            X_tensor = Tensor(X_norm, mstype.float32)
            y_pred_norm = self.net(X_tensor).asnumpy()
        
        # 反标准化
        y_pred = self._denormalize_y(y_pred_norm)
        return y_pred.flatten()


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("MindSpore风速预测模型测试")
    print("=" * 70)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_neighbors = 4
    
    # 训练数据（模拟风机之间的相关性）
    X_train = np.random.randn(n_samples, n_neighbors).astype(np.float32) * 5 + 10
    # 目标风机受邻近风机影响
    y_train = (0.3 * X_train[:, 0] + 0.25 * X_train[:, 1] + 
               0.25 * X_train[:, 2] + 0.2 * X_train[:, 3] + 
               np.random.randn(n_samples) * 0.5)
    
    # 测试数据
    X_test = np.random.randn(100, n_neighbors).astype(np.float32) * 5 + 10
    y_test = (0.3 * X_test[:, 0] + 0.25 * X_test[:, 1] + 
              0.25 * X_test[:, 2] + 0.2 * X_test[:, 3])
    
    print(f"\n训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    print(f"特征维度: {n_neighbors}")
    
    # 创建并训练模型
    print(f"\n{'='*70}")
    print("开始训练...")
    print(f"{'='*70}\n")
    
    model = MindSporeWindPredictor(n_neighbors=n_neighbors, hidden_size=64)
    
    if model.framework == 'mindspore':
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=True)
    else:
        model.fit(X_train, y_train, epochs=1, verbose=True)
    
    # 预测
    print(f"\n{'='*70}")
    print("开始预测...")
    print(f"{'='*70}\n")
    
    y_pred = model.predict(X_test)
    
    # 评估
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(mse)
    
    print("模型评估结果:")
    print(f"  MSE (均方误差):     {mse:.6f}")
    print(f"  RMSE (均方根误差):   {rmse:.6f}")
    print(f"  MAE (平均绝对误差):  {mae:.6f}")
    
    # 预测样例
    print(f"\n预测样例 (前5个):")
    print("-" * 70)
    for i in range(5):
        error = abs(y_test[i] - y_pred[i])
        error_pct = (error / y_test[i]) * 100 if y_test[i] != 0 else 0
        print(f"  样本{i+1}: 真实={y_test[i]:6.2f}, "
              f"预测={y_pred[i]:6.2f}, "
              f"误差={error:5.2f} ({error_pct:4.1f}%)")
    
    print(f"\n{'='*70}")
    print("✓ 测试完成！")
    print(f"{'='*70}")
