# mindspore_gnn_model.py
"""
基于MindSpore的时空图神经网络（ST-GNN）风速预测模型
用于风能发电机测量风速的误差校正

模型架构：
  - 时序特征提取：多层1D卷积 + 全局池化（高效替代LSTM）
  - 空间特征聚合：4层图卷积网络（GCN）
  - 注意力机制：自适应节点权重学习
  - 残差连接：提升深层网络训练稳定性
"""

import numpy as np

# 尝试导入MindSpore
try:
    import mindspore
    from mindspore import nn, ops, Tensor, context
    from mindspore import dtype as mstype
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
    """图卷积层：消息传递 + 特征变换"""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Dense(in_features, out_features, weight_init='xavier_uniform')
        self.activation = nn.ReLU()

    def construct(self, node_features, adjacency_matrix):
        aggregated = ops.matmul(adjacency_matrix, node_features)
        output = self.linear(aggregated)
        return self.activation(output)


# =========================================================================
# 2. 时序特征提取器（高效版）
# =========================================================================
class TemporalEncoder(nn.Cell):
    """
    时序特征编码器
    使用多层全连接网络提取时间序列特征（高效、CPU友好）
    """

    def __init__(self, seq_len, n_features=1, hidden_dim=64):
        super(TemporalEncoder, self).__init__()
        input_dim = seq_len * n_features
        self.flatten_dim = input_dim
        self.encoder = nn.SequentialCell([
            nn.Dense(input_dim, hidden_dim * 2, weight_init='xavier_uniform'),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Dense(hidden_dim * 2, hidden_dim, weight_init='xavier_uniform'),
            nn.ReLU(),
        ])

    def construct(self, x):
        # x: [seq_len, n_features] -> flatten -> encode
        flat = x.view(x.shape[0], -1) if len(x.shape) == 3 else x.view(-1)
        if len(flat.shape) == 1:
            flat = flat.view(1, -1)
        return self.encoder(flat)


# =========================================================================
# 3. 时空图神经网络
# =========================================================================
class SpatioTemporalGNN(nn.Cell):
    """
    时空图神经网络（Spatial-Temporal GNN）

    架构：
      时序编码 -> 图卷积(×4层) -> 注意力 -> 输出
      含残差连接和BatchNorm
    """

    def __init__(self, n_features=1, hidden_dim=128, n_layers=4, seq_len=24):
        super(SpatioTemporalGNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # 时序特征编码器
        self.temporal_encoder = TemporalEncoder(
            seq_len=seq_len, n_features=n_features, hidden_dim=hidden_dim // 2
        )

        # 多层图卷积
        self.graph_conv_layers = nn.CellList([
            GraphConvLayer(hidden_dim // 2 if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])

        # BatchNorm
        self.batch_norms = nn.CellList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])

        # 注意力机制
        self.attention = nn.Dense(hidden_dim, 1)

        # 残差投影
        self.residual_proj = nn.Dense(hidden_dim // 2, hidden_dim)

        # 输出层
        self.fc1 = nn.Dense(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Dense(hidden_dim // 2, 32)
        self.output_layer = nn.Dense(32, 1, weight_init='xavier_uniform')

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, node_temporal_features, adjacency_matrix, target_node_idx=None):
        """
        前向传播

        Args:
            node_temporal_features: [n_nodes, seq_len, n_features] 或 [n_nodes, n_features]
            adjacency_matrix: [n_nodes, n_nodes]
            target_node_idx: int or None

        Returns:
            prediction: [1, 1] or [n_nodes, 1]
        """
        # 1. 时序特征提取
        if len(node_temporal_features.shape) == 3:
            # [n_nodes, seq_len, n_features] -> 逐节点编码
            n_nodes = node_temporal_features.shape[0]
            encoded_list = []
            for i in range(n_nodes):
                node_seq = node_temporal_features[i]  # [seq_len, n_features]
                flat = node_seq.view(1, -1)  # [1, seq_len * n_features]
                enc = self.temporal_encoder.encoder(flat)  # [1, hidden_dim//2]
                encoded_list.append(enc)
            x = ops.Concat(axis=0)(encoded_list)  # [n_nodes, hidden_dim//2]
        elif len(node_temporal_features.shape) == 2:
            x = self.temporal_encoder(node_temporal_features)
        else:
            x = node_temporal_features

        # 残差基准
        residual = self.residual_proj(x)

        # 2. 多层图卷积（空间聚合）
        for i, conv_layer in enumerate(self.graph_conv_layers):
            x = conv_layer(x, adjacency_matrix)
            x = self.batch_norms[i](x)
            x = self.dropout(x)
            if i % 2 == 1 and i > 0:
                x = x + residual
                residual = x

        # 3. 注意力
        att_scores = self.attention(x)
        att_weights = self.sigmoid(att_scores)
        x = x * att_weights

        # 4. 输出
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = self.output_layer(x)

        if target_node_idx is not None:
            return output[target_node_idx:target_node_idx+1]
        return output


# =========================================================================
# 4. 封装类（训练/预测接口）
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
            self._build_model()
            self.framework = 'mindspore'
            print("  使用框架: MindSpore ST-GNN")

        self.mean_X = self.std_X = self.mean_y = self.std_y = None

    def _build_model(self):
        self.net = SpatioTemporalGNN(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            seq_len=self.seq_len
        )
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.net.trainable_params(), learning_rate=0.001)

    def _normalize(self, X, y=None, fit=False):
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
            return X_norm, (y - self.mean_y) / self.std_y
        return X_norm

    def _denormalize_y(self, y_norm):
        return y_norm * self.std_y + self.mean_y

    def _build_adjacency_matrix(self, n_nodes, positions=None):
        if positions is not None:
            from scipy.spatial.distance import cdist
            dist = cdist(positions, positions, metric='euclidean')
            sigma = np.median(dist) + 1e-8
            adj = np.exp(-dist ** 2 / (2 * sigma ** 2))
            np.fill_diagonal(adj, 0)
            deg = adj.sum(axis=1, keepdims=True)
            adj = adj / (deg + 1e-8)
        else:
            adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
            np.fill_diagonal(adj, 0)
            adj = adj / (n_nodes - 1)
        return adj.astype(np.float32)

    def fit(self, X, y, positions=None, epochs=50, batch_size=32, verbose=True):
        X_norm, y_norm = self._normalize(X, y, fit=True)

        if self.framework == 'sklearn':
            self.model.fit(X_norm.reshape(X_norm.shape[0], -1), y_norm.ravel())
            if verbose:
                print("  ✓ 训练完成")
            return

        n_samples = X_norm.shape[0]
        n_neighbors = X_norm.shape[1]
        adj = self._build_adjacency_matrix(n_neighbors, positions)
        adj_t = Tensor(adj, mstype.float32)

        self.net.set_train(True)

        def forward_fn(feat, a, label, t_idx):
            pred = self.net(feat, a, t_idx)
            return self.loss_fn(pred, label), pred

        grad_fn = mindspore.value_and_grad(
            forward_fn, None, self.optimizer.parameters, has_aux=True
        )

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss, cnt = 0.0, 0

            for idx in indices:
                if len(X_norm.shape) >= 3:
                    feat = Tensor(X_norm[idx], mstype.float32)
                else:
                    feat = Tensor(X_norm[idx].reshape(-1, 1), mstype.float32)
                label = Tensor(y_norm[idx].reshape(1, 1), mstype.float32)

                (loss, _), grads = grad_fn(feat, adj_t, label, 0)
                self.optimizer(grads)
                epoch_loss += loss.asnumpy()
                cnt += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/cnt:.6f}")

        if verbose:
            print("  ✓ 训练完成")

    def predict(self, X, positions=None):
        X_norm = self._normalize(X, fit=False)

        if self.framework == 'sklearn':
            return self._denormalize_y(
                self.model.predict(X_norm.reshape(X_norm.shape[0], -1)).reshape(-1, 1)
            ).flatten()

        self.net.set_train(False)
        n_neighbors = X_norm.shape[1]
        adj = self._build_adjacency_matrix(n_neighbors, positions)
        adj_t = Tensor(adj, mstype.float32)

        preds = []
        for i in range(X_norm.shape[0]):
            if len(X_norm.shape) >= 3:
                feat = Tensor(X_norm[i], mstype.float32)
            else:
                feat = Tensor(X_norm[i].reshape(-1, 1), mstype.float32)
            pred = self.net(feat, adj_t, target_node_idx=0)
            preds.append(pred.asnumpy())

        return self._denormalize_y(np.array(preds).reshape(-1, 1)).flatten()


# =========================================================================
# 5. 测试
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MindSpore时空图神经网络（ST-GNN）风速预测模型测试")
    print("=" * 70)

    np.random.seed(42)
    n_train, n_test, n_neighbors, seq_len = 100, 30, 4, 12

    positions = np.random.randn(n_neighbors, 2).astype(np.float32) * 10

    X_train = np.random.randn(n_train, n_neighbors, seq_len, 1).astype(np.float32) * 5 + 10
    y_train = (0.3 * X_train[:, 0, -1, 0] + 0.25 * X_train[:, 1, -1, 0] +
               0.25 * X_train[:, 2, -1, 0] + 0.2 * X_train[:, 3, -1, 0] +
               np.random.randn(n_train).astype(np.float32) * 0.5)

    X_test = np.random.randn(n_test, n_neighbors, seq_len, 1).astype(np.float32) * 5 + 10
    y_test = (0.3 * X_test[:, 0, -1, 0] + 0.25 * X_test[:, 1, -1, 0] +
              0.25 * X_test[:, 2, -1, 0] + 0.2 * X_test[:, 3, -1, 0])

    print(f"\n训练集: {n_train}样本, 测试集: {n_test}样本")
    print(f"风机数: {n_neighbors}, 时序: {seq_len}h")
    print(f"数据形状: {X_train.shape}")

    print(f"\n{'='*70}")
    print("训练ST-GNN...")
    print(f"{'='*70}\n")

    model = MindSporeGNNPredictor(n_features=1, hidden_dim=128, n_layers=4, seq_len=seq_len)
    model.fit(X_train, y_train, positions=positions, epochs=20, verbose=True)

    print(f"\n{'='*70}")
    print("预测中...")
    print(f"{'='*70}\n")

    y_pred = model.predict(X_test, positions=positions)

    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_test))

    print(f"MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    print(f"\n预测样例:")
    print("-" * 60)
    for i in range(min(5, n_test)):
        err = abs(y_test[i] - y_pred[i])
        pct = err / abs(y_test[i]) * 100 if y_test[i] != 0 else 0
        print(f"  #{i+1}: 真实={y_test[i]:7.2f} 预测={y_pred[i]:7.2f} 误差={err:.2f}({pct:.1f}%)")

    print(f"\n{'='*70}")
    print("✓ ST-GNN模型测试完成！")
    print("  架构: 时序编码器 + 4层图卷积 + 注意力 + 残差连接")
    print("  参数: ~50k+")
    print("=" * 70)
