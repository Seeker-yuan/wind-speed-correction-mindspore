# mindspore_gnn_model.py
"""
基于MindSpore的时空图神经网络(ST-GNN)风速预测模型
用于风能发电机测量风速的误差校正

模型架构:
  - 图卷积层(GraphConvLayer): 消息传递 + 特征变换
  - 时空图神经网络(SpatioTemporalGNN): 时序编码 + 4层GCN + 注意力 + 残差
  - 兼容预测器(MindSporeWindPredictor): 图神经网络版预测接口
"""

import numpy as np

# =========================================================================
# MindSpore / sklearn 自适应导入
# =========================================================================
try:
    import mindspore
    from mindspore import nn, ops, Tensor, context
    from mindspore import dtype as mstype
    from mindspore.nn import MSELoss, Adam
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    MINDSPORE_AVAILABLE = True
    print("[INFO] MindSpore ST-GNN (CPU)")
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("[INFO] MindSpore unavailable, sklearn fallback")
    from sklearn.neural_network import MLPRegressor


# =========================================================================
# 1. 图卷积层 (Graph Convolution Layer)
# =========================================================================
if MINDSPORE_AVAILABLE:
    class GraphConvLayer(nn.Cell):
        """
        图卷积层: A * H -> Linear -> ReLU
        实现消息传递机制，聚合邻居节点信息
        """
        def __init__(self, in_features, out_features):
            super(GraphConvLayer, self).__init__()
            self.linear = nn.Dense(in_features, out_features,
                                   weight_init='xavier_uniform')
            self.activation = nn.ReLU()

        def construct(self, node_features, adjacency_matrix):
            # 消息传递: 邻居特征聚合
            aggregated = ops.matmul(adjacency_matrix, node_features)
            # 特征变换
            output = self.linear(aggregated)
            return self.activation(output)


# =========================================================================
# 2. 时空图神经网络 (Spatio-Temporal GNN)
# =========================================================================
if MINDSPORE_AVAILABLE:
    class SpatioTemporalGNN(nn.Cell):
        """
        时空图神经网络

        架构:
            输入投影 -> 4层图卷积(含残差+BatchNorm) -> 注意力 -> 输出MLP
        """
        def __init__(self, input_dim=1, hidden_dim=64, n_layers=4):
            super(SpatioTemporalGNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim

            # 输入投影
            self.input_proj = nn.Dense(input_dim, hidden_dim,
                                       weight_init='xavier_uniform')

            # 多层图卷积
            self.gcn_layers = nn.CellList([
                GraphConvLayer(hidden_dim, hidden_dim)
                for _ in range(n_layers)
            ])

            # 残差投影 (当维度变化时)
            self.residual_proj = nn.Dense(hidden_dim, hidden_dim,
                                          weight_init='xavier_uniform')

            # 注意力机制
            self.attention_fc = nn.Dense(hidden_dim, 1)

            # 输出MLP
            self.fc1 = nn.Dense(hidden_dim, hidden_dim // 2,
                                weight_init='xavier_uniform')
            self.fc2 = nn.Dense(hidden_dim // 2, 16,
                                weight_init='xavier_uniform')
            self.output_layer = nn.Dense(16, 1,
                                         weight_init='xavier_uniform')

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.2)

        def construct(self, node_features, adjacency_matrix, target_idx=None):
            """
            Args:
                node_features: [n_nodes, input_dim]
                adjacency_matrix: [n_nodes, n_nodes]
                target_idx: int or None
            Returns:
                [1, 1] or [n_nodes, 1]
            """
            # 输入投影
            x = self.input_proj(node_features)      # [n_nodes, hidden_dim]
            residual = self.residual_proj(x)

            # 多层GCN + 残差
            for i in range(self.n_layers):
                x = self.gcn_layers[i](x, adjacency_matrix)
                x = self.dropout(x)
                if i % 2 == 1:
                    x = x + residual
                    residual = x

            # 注意力加权
            att_scores = self.sigmoid(self.attention_fc(x))  # [n_nodes, 1]
            x = x * att_scores

            # 输出MLP
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            out = self.output_layer(x)  # [n_nodes, 1]

            if target_idx is not None:
                return out[target_idx:target_idx + 1]
            return out


# =========================================================================
# 3. 预测器封装 (兼容管道接口)
# =========================================================================
class MindSporeWindPredictor:
    """
    基于图神经网络的风速预测器

    接口兼容:
        model = MindSporeWindPredictor(n_neighbors=4, hidden_size=64)
        model.fit(X, y, epochs=30, batch_size=32, verbose=False)
        preds = model.predict(X_test)

    其中 X shape = (n_samples, n_neighbors, seq_len), 每个样本是 n 台候选风机过去 seq_len 小时的风速
    """

    def __init__(self, n_neighbors=4, hidden_size=64, seq_len=6):
        self.n_neighbors = n_neighbors
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.mean_X = self.std_X = self.mean_y = self.std_y = None

        if MINDSPORE_AVAILABLE:
            self.net = SpatioTemporalGNN(
                input_dim=seq_len,   # 每个节点的特征维度 = 时序窗口长度
                hidden_dim=hidden_size,
                n_layers=4
            )
            self.loss_fn = MSELoss()
            self.optimizer = Adam(self.net.trainable_params(),
                                  learning_rate=0.001)
            self.framework = 'mindspore'
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden_size, hidden_size, 32),
                activation='relu', solver='adam',
                max_iter=200, random_state=42
            )
            self.framework = 'sklearn'

    def _normalize(self, X, y=None, fit=False):
        X = np.array(X, dtype=np.float32)
        if fit:
            self.mean_X = X.mean(axis=0, keepdims=True)
            self.std_X = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - self.mean_X) / self.std_X
        if y is not None:
            y = np.array(y, dtype=np.float32).reshape(-1)
            if fit:
                self.mean_y = y.mean()
                self.std_y = y.std() + 1e-8
            return X_norm, (y - self.mean_y) / self.std_y
        return X_norm

    def _denormalize_y(self, y_norm):
        return y_norm * self.std_y + self.mean_y

    def _build_adj(self, n, positions=None):
        """构建邻接矩阵"""
        if positions is not None:
            from scipy.spatial.distance import cdist
            dist = cdist(positions, positions, metric='euclidean')
            sigma = np.median(dist) + 1e-8
            adj = np.exp(-dist ** 2 / (2 * sigma ** 2))
        else:
            adj = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(adj, 0)
        deg = adj.sum(axis=1, keepdims=True) + 1e-8
        adj = adj / deg
        return adj.astype(np.float32)

    def fit(self, X, y, epochs=30, batch_size=32, verbose=True, positions=None):
        """
        训练模型

        Args:
            X: (n_samples, n_neighbors, seq_len) 候选风机时序特征
            y: (n_samples,) 目标风机风速
            epochs: 训练轮数
            batch_size: 批大小 (未使用, 保持接口一致)
            verbose: 是否输出训练日志
            positions: 可选, 风机坐标用于构建邻接矩阵
        """
        X_norm, y_norm = self._normalize(X, y, fit=True)

        if self.framework == 'sklearn':
            # sklearn 不支持 3D 输入，需展平为 (n_samples, n_neighbors*seq_len)
            self.model.fit(X_norm.reshape(X_norm.shape[0], -1), y_norm.ravel())
            if verbose:
                print("  [OK] sklearn training done")
            return

        n_samples = X_norm.shape[0]
        n_nodes = X_norm.shape[1]  # n_neighbors
        adj = self._build_adj(n_nodes, positions)
        adj_t = Tensor(adj, mstype.float32)

        self.net.set_train(True)

        def forward_fn(feat, a, label):
            pred = self.net(feat, a, target_idx=0)
            return self.loss_fn(pred, label), pred

        grad_fn = mindspore.value_and_grad(
            forward_fn, None, self.optimizer.parameters, has_aux=True
        )

        best_loss = float('inf')
        patience_cnt = 0
        PATIENCE = 3  # 连续3轮无改善则提前停止

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for idx in indices:
                # 每个样本: shape (n_neighbors, seq_len) — 节点数 × 时序特征维度
                feat = Tensor(X_norm[idx], mstype.float32)
                label = Tensor(
                    np.array([[y_norm[idx]]], dtype=np.float32), mstype.float32
                )

                (loss, _), grads = grad_fn(feat, adj_t, label)
                self.optimizer(grads)
                epoch_loss += loss.asnumpy()

            avg = epoch_loss / n_samples
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")

            # 早停：连续 PATIENCE 轮损失无改善则中止
            if avg < best_loss - 1e-6:
                best_loss = avg
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    if verbose:
                        print(f"    Early stop at epoch {epoch+1} (loss={avg:.6f})")
                    break

        if verbose:
            print("  [OK] ST-GNN training done")

    def predict(self, X, positions=None):
        """
        预测

        Args:
            X: (n_pred, n_neighbors, seq_len) 候选风机时序特征
        Returns:
            numpy array of predictions
        """
        X_norm = self._normalize(X, fit=False)

        if self.framework == 'sklearn':
            pred_norm = self.model.predict(X_norm.reshape(X_norm.shape[0], -1)).reshape(-1)
            return np.clip(self._denormalize_y(pred_norm), 0.0, None)

        self.net.set_train(False)
        n_nodes = X_norm.shape[1]
        adj = self._build_adj(n_nodes, positions)
        adj_t = Tensor(adj, mstype.float32)

        preds = []
        for i in range(X_norm.shape[0]):
            feat = Tensor(X_norm[i], mstype.float32)  # (n_neighbors, seq_len)
            pred = self.net(feat, adj_t, target_idx=0)
            preds.append(pred.asnumpy().item())

        # 风速物理约束：不能为负
        return np.clip(self._denormalize_y(np.array(preds, dtype=np.float32)), 0.0, None)


# 向后兼容别名
MindSporeGNNPredictor = MindSporeWindPredictor
GNNWindPredictor = MindSporeWindPredictor


# =========================================================================
# 4. 测试
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MindSpore ST-GNN Wind Speed Prediction Test")
    print("=" * 60)

    np.random.seed(42)
    N_TRAIN, N_TEST, N_NEIGHBORS, SEQ_LEN = 40, 15, 4, 6

    # 模拟数据: (n_samples, n_neighbors, seq_len) — 每台候选风机过去6小时风速
    X_train = np.random.randn(N_TRAIN, N_NEIGHBORS, SEQ_LEN).astype(np.float32) * 3 + 8
    y_train = (0.3 * X_train[:, 0, -1] + 0.25 * X_train[:, 1, -1] +
               0.25 * X_train[:, 2, -1] + 0.2 * X_train[:, 3, -1] +
               np.random.randn(N_TRAIN).astype(np.float32) * 0.3)

    X_test = np.random.randn(N_TEST, N_NEIGHBORS, SEQ_LEN).astype(np.float32) * 3 + 8
    y_test = (0.3 * X_test[:, 0, -1] + 0.25 * X_test[:, 1, -1] +
              0.25 * X_test[:, 2, -1] + 0.2 * X_test[:, 3, -1])

    print(f"\nTrain: {N_TRAIN}, Test: {N_TEST}, Neighbors: {N_NEIGHBORS}, seq_len: {SEQ_LEN}")
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
    print()

    model = MindSporeWindPredictor(n_neighbors=N_NEIGHBORS, hidden_size=64, seq_len=SEQ_LEN)
    model.fit(X_train, y_train, epochs=20, verbose=True)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"\nRMSE={rmse:.4f}, MAE={mae:.4f}")

    print(f"\nSample predictions:")
    for i in range(min(5, N_TEST)):
        err = abs(y_test[i] - y_pred[i])
        print(f"  True={y_test[i]:.2f}  Pred={y_pred[i]:.2f}  Err={err:.2f}")

    print(f"\n{'='*60}")
    print("Architecture: 4-layer GCN + Attention + Residual + Dropout")
    print("=" * 60)
