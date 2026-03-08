# mindspore_gnn_model.py
"""
基于MindSpore的时空图神经网络（ST-GNN）风速预测模型
用于风能发电机测量风速的误差校正

模型架构：
  - 2层LSTM：捕获时序特征
  - 4层图卷积：建模空间拓扑关系
  - 注意力机制：自适应节点权重
  - 残差连接：提升深层网络训练
"""

import numpy as np

# 尝试导入MindSpore
try:
    import mindspore
    from mindspore import nn, ops, Tensor, context
    from mindspore import dtype as mstype
    from mindspore.train import Model
    from mindspore.nn import MSELoss, Adam

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    MINDSPORE_AVAILABLE = True
    print("✓ MindSpore已加载 (CPU模式)")
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("⚠ MindSpore未安装，使用sklearn作为后备")
    from sklearn.neural_network import MLPRegressor


# =========================================================================
# 1. 图卷积层
# =========================================================================
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
        # 消息传递：A * H
        aggregated = ops.matmul(adjacency_matrix, node_features)
        output = self.linear(aggregated)
        output = self.activation(output)
        return output


# =========================================================================
# 2. 时空图神经网络
# =========================================================================
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

        # 多层图卷积
        self.graph_conv_layers = nn.CellList([
            GraphConvLayer(hidden_dim // 2 if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])

        # 注意力机制
        self.attention = nn.Dense(hidden_dim, 1)

        # 残差连接的投影层
        self.residual_proj = nn.Dense(hidden_dim // 2, hidden_dim)

        # 输出层
        self.fc1 = nn.Dense(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Dense(hidden_dim // 2, 32)
        self.output_layer = nn.Dense(32, 1, weight_init='xavier_uniform')

        # 正则化
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norms = nn.CellList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
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

        # 3. 注意力机制（突出重要节点）
        attention_scores = self.attention(x)
        attention_weights = self.sigmoid(attention_scores)
        x = x * attention_weights

        # 4. 深度输出层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        output = self.output_layer(x)

        # 如果指定了目标节点，只返回该节点的预测
        if target_node_idx is not None:
            return output[target_node_idx:target_node_idx+1]

        return output


# =========================================================================
# 3. 封装类（训练/预测）
# =========================================================================
class MindSporeGNNPredictor:
    """MindSpore时空图神经网络预测器"""

    def __init__(self, n_features=1, hidden_dim=128, n_layers=4, seq_len=24):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_len = seq_len

        if not MINDSPORE_AVAILABLE:
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, hidden_dim, 32),
                activation='relu', solver='adam',
                max_iter=200, random_state=42
            )
            self.framework = 'sklearn'
            print("  使用框架: sklearn (后备)")
        else:
            self._build_mindspore_model()
            self.framework = 'mindspore'
            print("  使用框架: MindSpore GNN")

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
        """构建邻接矩阵（基于地理距离的高斯核）"""
        if positions is not None:
            from scipy.spatial.distance import cdist
            distances = cdist(positions, positions, metric='euclidean')
            sigma = np.median(distances) + 1e-8
            adjacency = np.exp(-distances ** 2 / (2 * sigma ** 2))
            np.fill_diagonal(adjacency, 0)
            degree = adjacency.sum(axis=1, keepdims=True)
            adjacency = adjacency / (degree + 1e-8)
        else:
            adjacency = np.ones((n_nodes, n_nodes), dtype=np.float32)
            np.fill_diagonal(adjacency, 0)
            adjacency = adjacency / (n_nodes - 1)
        return adjacency.astype(np.float32)

    def fit(self, X, y, positions=None, epochs=50, batch_size=32, verbose=True):
        """训练模型"""
        X_norm, y_norm = self._normalize(X, y, fit=True)

        if self.framework == 'sklearn':
            X_flat = X_norm.reshape(X_norm.shape[0], -1)
            self.model.fit(X_flat, y_norm.ravel())
            if verbose:
                print("  ✓ 训练完成")
            return

        # MindSpore训练
        n_samples = X_norm.shape[0]
        n_neighbors = X_norm.shape[1]

        adjacency = self._build_adjacency_matrix(n_neighbors, positions)
        adjacency_tensor = Tensor(adjacency, mstype.float32)

        self.net.set_train(True)

        def forward_fn(node_feat, adj, label, t_idx):
            pred = self.net(node_feat, adj, t_idx)
            loss = self.loss_fn(pred, label)
            return loss, pred

        grad_fn = mindspore.value_and_grad(
            forward_fn, None, self.optimizer.parameters, has_aux=True
        )

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:min(i + batch_size, n_samples)]

                for idx in batch_idx:
                    # 准备单个样本的节点特征
                    if len(X_norm.shape) == 4:
                        # [n_neighbors, seq_len, n_features]
                        node_features = Tensor(X_norm[idx], mstype.float32)
                    elif len(X_norm.shape) == 3:
                        node_features = Tensor(X_norm[idx], mstype.float32)
                    else:
                        # [n_neighbors] -> [n_neighbors, 1]
                        node_features = Tensor(
                            X_norm[idx].reshape(-1, 1), mstype.float32
                        )

                    label = Tensor(y_norm[idx].reshape(1, 1), mstype.float32)

                    (loss, _), grads = grad_fn(
                        node_features, adjacency_tensor, label, 0
                    )
                    self.optimizer(grads)

                    epoch_loss += loss.asnumpy()
                    n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if verbose:
            print("  ✓ 训练完成")

    def predict(self, X, positions=None):
        """预测"""
        X_norm = self._normalize(X, fit=False)

        if self.framework == 'sklearn':
            X_flat = X_norm.reshape(X_norm.shape[0], -1)
            y_pred_norm = self.model.predict(X_flat).reshape(-1, 1)
        else:
            self.net.set_train(False)
            n_samples = X_norm.shape[0]
            n_neighbors = X_norm.shape[1]

            adjacency = self._build_adjacency_matrix(n_neighbors, positions)
            adjacency_tensor = Tensor(adjacency, mstype.float32)

            predictions = []
            for i in range(n_samples):
                if len(X_norm.shape) == 4 or len(X_norm.shape) == 3:
                    node_features = Tensor(X_norm[i], mstype.float32)
                else:
                    node_features = Tensor(
                        X_norm[i].reshape(-1, 1), mstype.float32
                    )
                pred = self.net(node_features, adjacency_tensor, target_node_idx=0)
                predictions.append(pred.asnumpy())

            y_pred_norm = np.array(predictions).reshape(-1, 1)

        y_pred = self._denormalize_y(y_pred_norm)
        return y_pred.flatten()


# =========================================================================
# 4. 测试代码
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MindSpore时空图神经网络（ST-GNN）风速预测模型测试")
    print("=" * 70)

    np.random.seed(42)

    # 使用较小数据集快速验证模型
    n_samples = 200
    n_neighbors = 4
    seq_len = 12  # 12小时历史窗口（测试用，实际可用24）

    # 模拟风机位置
    positions = np.random.randn(n_neighbors, 2).astype(np.float32) * 10

    # 训练数据 [n_samples, n_neighbors, seq_len, 1]
    X_train = np.random.randn(n_samples, n_neighbors, seq_len, 1).astype(np.float32) * 5 + 10
    y_train = (0.3 * X_train[:, 0, -1, 0] + 0.25 * X_train[:, 1, -1, 0] +
               0.25 * X_train[:, 2, -1, 0] + 0.2 * X_train[:, 3, -1, 0] +
               np.random.randn(n_samples).astype(np.float32) * 0.5)

    # 测试数据
    X_test = np.random.randn(50, n_neighbors, seq_len, 1).astype(np.float32) * 5 + 10
    y_test = (0.3 * X_test[:, 0, -1, 0] + 0.25 * X_test[:, 1, -1, 0] +
              0.25 * X_test[:, 2, -1, 0] + 0.2 * X_test[:, 3, -1, 0])

    print(f"\n训练样本数: {n_samples}")
    print(f"测试样本数: {len(X_test)}")
    print(f"邻近风机数: {n_neighbors}")
    print(f"时序长度: {seq_len}小时")
    print(f"训练数据形状: {X_train.shape}")

    print(f"\n{'='*70}")
    print("开始训练时空图神经网络（ST-GNN）...")
    print(f"{'='*70}\n")

    model = MindSporeGNNPredictor(
        n_features=1, hidden_dim=128, n_layers=4, seq_len=seq_len
    )

    model.fit(
        X_train, y_train, positions=positions,
        epochs=30, batch_size=16, verbose=True
    )

    print(f"\n{'='*70}")
    print("开始预测...")
    print(f"{'='*70}\n")

    y_pred = model.predict(X_test, positions=positions)

    # 评估
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(mse)

    print("模型评估结果:")
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    print(f"\n预测样例 (前5个):")
    print("-" * 70)
    for i in range(5):
        error = abs(y_test[i] - y_pred[i])
        error_pct = (error / abs(y_test[i])) * 100 if y_test[i] != 0 else 0
        print(f"  样本{i+1}: 真实={y_test[i]:8.2f}, "
              f"预测={y_pred[i]:8.2f}, "
              f"误差={error:6.2f} ({error_pct:4.1f}%)")

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
