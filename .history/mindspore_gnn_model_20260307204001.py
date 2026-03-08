# mindspore_gnn_model.py
"""
基于MindSpore的图神经网络（GNN）风速预测模型
用于风能发电机测量风速的误差校正
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


class GraphConvLayer(nn.Cell):
    """
    图卷积层 (Graph Convolution Layer)
    实现消息传递机制，聚合邻居节点信息
    """
    
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Dense(in_features, out_features, weight_init='xavier_uniform')
        self.activation = nn.ReLU()
    
    def construct(self, node_features, adjacency_matrix):
        """
        前向传播
        
        Args:
            node_features: [n_nodes, in_features] 节点特征
            adjacency_matrix: [n_nodes, n_nodes] 邻接矩阵
            
        Returns:
            aggregated_features: [n_nodes, out_features] 聚合后的特征
        """
        # 消息传递：聚合邻居节点信息
        # A * H: 邻接矩阵与节点特征相乘，实现信息聚合
        aggregated = ops.matmul(adjacency_matrix, node_features)
        
        # 特征变换
        output = self.linear(aggregated)
        output = self.activation(output)
        
        return output


class SpatioTemporalGNN(nn.Cell):
    """
    时空图神经网络（Spatial-Temporal GNN）
    结合LSTM捕获时序特征 + 图卷积捕获空间关系
    """
    
    def __init__(self, n_features=1, hidden_dim=128, n_layers=4, seq_len=24):
        super(SpatioTemporalGNN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 时序特征提取器（LSTM）
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 多层图卷积（更深的网络）
        self.graph_conv_layers = nn.CellList([
            GraphConvLayer(hidden_dim // 2 if i == 0 else hidden_dim, hidden_dim) 
            for i in range(n_layers)
        ])
        
        # 注意力机制
        self.attention = nn.Dense(hidden_dim, 1)
        
        # 残差连接的投影层
        self.residual_proj = nn.Dense(hidden_dim // 2, hidden_dim)
        
        # 输出层（更深）
        self.fc1 = nn.Dense(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Dense(hidden_dim // 2, 32)
        self.output_layer = nn.Dense(32, 1, weight_init='xavier_uniform')
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(p=0.3)
        
        # Batch Normalization（每层都有）
        self.batch_norms = nn.CellList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def construct(self, node_temporal_features, adjacency_matrix, target_node_idx=None):
        """
        前向传播
        
        Args:
            node_temporal_features: [n_nodes, seq_len, n_features] 节点的时序特征
            adjacency_matrix: [n_nodes, n_nodes] 图的邻接矩阵
            target_node_idx: int or None 目标节点索引
            
        Returns:
            prediction: [1] or [n_nodes, 1] 预测值
        """
        n_nodes = node_temporal_features.shape[0]
        
        # 1. 时序特征提取（对每个节点应用LSTM）
        temporal_features = []
        for i in range(n_nodes):
            node_seq = node_temporal_features[i:i+1]  # [1, seq_len, n_features]
            lstm_out, _ = self.lstm(node_seq)
            temporal_features.append(lstm_out[:, -1, :])  # 取最后时间步
        
        concat_op = ops.Concat(axis=0)
        x = concat_op(temporal_features)  # [n_nodes, hidden_dim//2]
        
        # 保存用于残差连接
        residual = self.residual_proj(x)
        
        # 2. 多层图卷积（空间特征聚合）
        for i, conv_layer in enumerate(self.graph_conv_layers):
            x = conv_layer(x, adjacency_matrix)
            x = self.batch_norms[i](x)
            x = self.dropout(x)
            
            # 残差连接（每2层添加一次）
            if i % 2 == 1 and i > 0:
                x = x + residual
                residual = x
        
        # 3. 注意力机制（突出重要节点）128, n_layers=4, seq_len=24):
        """
        初始化时空图神经网络模型
        
        Args:
            n_features: 每个节点的特征维度（风速数据为1）
            hidden_dim: 隐藏层维度（增加到128）
            n_layers: 图卷积层数（增加到4层）
            seq_len: 时序长度（历史窗口）
        """
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        output = self.output_layer(x)
        
        # 如果指定了目标节点，只返回该节点的预测
        if target_node_idx is not None:
            return output[target_node_idx:target_node_idx+1]
        
        return output


class GNNWindPredictor(nn.Cell):
    """向后兼容的别名"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SpatioTemporalGNN(*args, **kwargs)
    
    def construct(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MindSporeGNNPredictor:
    """
    MindSpore时空图神经网络预测器（封装类）
    """
    
    def __init__(self, n_features=1, hidden_dim=128, n_layers=4, seq_len=24):
        """
        初始化时空图神经网络模型
        
        Args:
            n_features: 每个节点的特征维度（风速数据为1）
            hidden_dim: 隐藏层维度（增加到128）
            n_layers: 图卷积层数（增加到4层）
            seq_len: 时序长度（历史窗口）
        """
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        if not MINDSPORE_AVAILABLE:
            # 后备方案：使用sklearn
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, hidden_dim, 32),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
            self.framework = 'sklearn'
            print("  使用框架: sklearn (后备)")
        else:
            # 使用MindSpore GNN
            self._build_mindspore_model()
            self.framework = 'mindspore'
            print("  使用框架: MindSpore GNN")
        
        # 标准化参数
        self.mean_X = None
        self.std_X = None
        self.mean_y = None
        self.std_y = None
    
    def _build_mindspore_model(self):
        """构建MindSpore时空图神经网络模型"""
        self.net = SpatioTemporalGNN(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            seq_len=self.seq_len
        )
        
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
    
    def _build_adjacency_matrix(self, n_nodes, positions=None):
        """
        构建邻接矩阵
        
        Args:
            n_nodes: 节点数量
            positions: 节点位置信息 [n_nodes, 2] (lon, lat)
            
        Returns:
            adjacency_matrix: [n_nodes, n_nodes] 邻接矩阵
        """
        if positions is not None:
            # 基于地理距离构建邻接矩阵
            from scipy.spatial.distance import cdist
            distances = cdist(positions, positions, metric='euclidean')
            
            # 使用高斯核转换距离为相似度
            sigma = np.median(distances)
            adjacency = np.exp(-distances ** 2 / (2 * sigma ** 2))
            
            # 对角线设为0（节点不连接自己）
            np.fill_diagonal(adjacency, 0)
            
            # 归一化（度归一化）
            degree = adjacency.sum(axis=1, keepdims=True)
            adjacency = adjacency / (degree + 1e-8)
        else:
            # 全连接图（所有节点相互连接）
            adjacency = np.ones((n_nodes, n_nodes), dtype=np.float32)
            np.fill_diagonal(adjacency, 0)
            adjacency = adjacency / (n_nodes - 1)
        
        return adjacency.astype(np.float32)
    
    def fit(self, X, y, positions=None, epochs=50, batch_size=32, verbose=True):
        """
        训练GNN模型
        
        Args:
            X: 输入特征 [n_samples, n_neighbors] 或 [n_samples, n_neighbors, n_features]
            y: 目标值 [n_samples]
            positions: 节点位置信息 [n_neighbors, 2]
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程
        """
        # 数据标准化
        X_norm, y_norm = self._normalize(X, y, fit=True)
        
        if self.framework == 'sklearn':
            # sklearn训练
            if len(X_norm.shape) == 3:
                X_norm = X_norm.reshape(X_norm.shape[0], -1)
            self.model.fit(X_norm, y_norm.ravel())
            if verbose:
                print("  ✓ 训练完成")
            return
        
        # MindSpore GNN训练
        n_samples = X_norm.shape[0]
        n_neighbors = X_norm.shape[1] if len(X_norm.shape) >= 2 else 1
        
        # 构建邻接矩阵
        adjacency = self._build_adjacency_matrix(n_neighbors, positions)
        adjacency_tensor = Tensor(adjacency, mstype.float32)
        
        self.net.set_train(True)
        
        # 定义前向函数
        def forward_fn(node_feat, adj, labels, target_idx):
            pred = self.net(node_feat, adj, target_idx)
            loss = self.loss_fn(pred, labels)
            return loss, pred
        
        # 获取梯度函数
        grad_fn = mindspore.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)
        
        # 训练循环
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_idx = indices[i:end_idx]
                
                # 准备批次数据
                if len(X_norm.shape) == 2:
                    batch_X = X_norm[batch_idx]  # [batch_size, n_neighbors]
                    # 重塑为 [n_neighbors, 1]（每个邻居是一个节点）
                    node_features = Tensor(batch_X[0].reshape(-1, 1), mstype.float32)
                else:
                    node_features = Tensor(X_norm[batch_idx[0]], mstype.float32)
                
                batch_y = Tensor(y_norm[batch_idx], mstype.float32)
                
                # 目标节点索引（假设预测第一个节点）
                target_idx = 0
                
                # 前向传播和反向传播
                (loss, _), grads = grad_fn(node_features, adjacency_tensor, batch_y, target_idx)
                
                # 更新参数
                self.optimizer(grads)
                
                epoch_loss += loss.asnumpy()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if verbose:
            print("  ✓ 训练完成")
    
    def predict(self, X, positions=None):
        """
        预测
        
        Args:
            X: 输入特征
            positions: 节点位置信息
            
        Returns:
            predictions: 预测值
        """
        X_norm = self._normalize(X, fit=False)
        
        if self.framework == 'sklearn':
            if len(X_norm.shape) == 3:
                X_norm = X_norm.reshape(X_norm.shape[0], -1)
            y_pred_norm = self.model.predict(X_norm).reshape(-1, 1)
        else:
            # MindSpore GNN预测
            self.net.set_train(False)
            
            n_samples = X_norm.shape[0]
            predictions = []
            
            n_neighbors = X_norm.shape[1] if len(X_norm.shape) >= 2 else 1
            adjacency = self._build_adjacency_matrix(n_neighbors, positions)
            adjacency_tensor = Tensor(adjacency, mstype.float32)
            
            for i in range(n_samples):
                if len(X_norm.shape) == 2:
                    node_features = Tensor(X_norm[i].reshape(-1, 1), mstype.float32)
                else:
                    node_features = Tensor(X_norm[i], mstype.float32)
                
                pred = self.net(node_features, adjacency_tensor, target_node_idx=0)
                predictions.append(pred.asnumpy())
            
            y_pred_norm = np.array(predictions).reshape(-1, 1)
        
        # 反标准化
        y_pred = self._denormalize_y(y_pred_norm)
        return y_pred.flatten()


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("MindSpore时空图神经网络（ST-GNN）风速预测模型测试")
    print("=" * 70)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_neighbors = 4  # 4个邻近风机
    
    # 模拟风机位置
    positions = np.random.randn(n_neighbors, 2).astype(np.float32) * 10
    
    # 训练数据（邻近风机风速）
    X_train = np.random.randn(n_samples, n_neighbors).astype(np.float32) * 5 + 10
    # 目标风机受邻近风机影响（考虑空间关系）
    y_train = (0.3 * X_train[:, 0] + 0.25 * X_train[:, 1] + 
               0.25 * X_train[:, 2] + 0.2 * X_train[:, 3] + 
               np.random.randn(n_samples) * 0.5)
    
    # 测试数据
    X_test = np.random.randn(100, n_neighbors).astype(np.float32) * 5 + 10
    y_test = (0.3 * X_test[:, 0] + 0.25 * X_test[:, 1] + 
              0.25 * X_test[:, 2] + 0.2 * X_test[:, 3])
    
    print(f"\n训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    print(f"邻近风机数: {n_neighbors}")
    print(f"图节点数: {n_neighbors} (每个风机是一个节点)")
    
    # 创建并训练时空GNN模型
    print(f"\n{'='*70}")
    print("开始训练时空图神经网络（ST-GNN）...")
    print(f"{'='*70}\n")
    
    model = MindSporeGNNPredictor(n_features=1, hidden_dim=128, n_layers=4, seq_len=24)
    
    if model.framework == 'mindspore':
        model.fit(X_train, y_train, positions=positions, epochs=50, batch_size=32, verbose=True)
    else:
        model.fit(X_train, y_train, epochs=1, verbose=True)
    
    # 预测
    print(f"\n{'='*70}")
    print("开始预测...")
    print(f"{'='*70}\n")
    
    y_pred = model.predict(X_test, positions=positions)
    
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
    print("\n时空图神经网络（ST-GNN）优势:")
    print("  1. LSTM捕获时序依赖关系")
    print("  2. 图卷积建模空间拓扑结构")
    print("  3. 注意力机制突出重要节点")
    print("  4. 残差连接提升深层网络训练")
    print("  5. 4层GNN + 2层LSTM，模型更深更强")
    print("=" * 70)
