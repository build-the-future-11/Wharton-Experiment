"""M08 Dynamic Graph Financial JEPA."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m08.config import M08Config
from wharton_lab.models.m08.graph import (
    adjacency_from_correlation,
    shrinkage_correlation,
    shuffle_edges_control,
)
from wharton_lab.models.torch_utils import pick_device, require_torch, set_deterministic_seed

torch = require_torch()
nn = torch.nn


class _TemporalNodeEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(node_dim, hidden, batch_first=True)

    def forward(self, x, mask):
        # x: (batch, nodes, time, feat) -> encode each node
        b, n, t, f = x.shape
        x = x.reshape(b * n, t, f)
        out, h = self.gru(x)
        h = h.squeeze(0).reshape(b, n, -1)
        return h * mask.unsqueeze(-1)


class _GraphMessagePassing(nn.Module):
    def __init__(self, hidden: int, passes: int) -> None:
        super().__init__()
        self.passes = passes
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, h, adj):
        # h (b, n, d), adj (n, n)
        for _ in range(self.passes):
            m = torch.einsum("ij,bjd->bid", adj, h)
            h = torch.relu(self.lin(h + m))
        return h


class _FutureLatentHead(nn.Module):
    def __init__(self, hidden: int, latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, latent),
            nn.ReLU(),
            nn.Linear(latent, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class GraphFIJEPAModel(BaseModel):
    MODEL_ID: ClassVar[str] = "M08"
    metadata: ClassVar[ModelMetadata] = ModelMetadata(
        model_id="M08",
        title="Dynamic Graph Financial JEPA",
        version="0.1.0",
        description="Shrinkage-correlation graph, temporal node encoder, message passing, future latent.",
        inputs=("node_sequences",),
        outputs=("node_forecasts",),
        training_constraints=("freeze_graph_before_predict",),
        metrics=("graph_shrinkage", "masked_nodes"),
        tags=("jepa", "graph", "financial"),
    )

    def __init__(self, config: M08Config | None = None) -> None:
        self.config = config or M08Config()
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        h = self.config.hidden_dim
        self.encoder = _TemporalNodeEncoder(self.config.node_dim, h).to(self.device)
        self.mp = _GraphMessagePassing(h, self.config.message_passes).to(self.device)
        self.head = _FutureLatentHead(h, self.config.latent_dim).to(self.device)
        self._opt = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.mp.parameters())
            + list(self.head.parameters()),
            lr=self.config.lr,
        )
        self._frozen_adj: np.ndarray | None = None
        self._rng = np.random.default_rng(self.config.random_state)
        self._fitted = False

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_factors=True)

    def config_dict(self) -> Mapping[str, Any]:
        return self.config.to_dict()

    def _parse_X(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tensor (n, nodes, seq, feat) and node mask (n, nodes)."""
        X = np.asarray(X, dtype=float)
        n_nodes = self.config.n_nodes
        sl = self.config.seq_len
        fd = self.config.node_dim
        if X.ndim == 4:
            return X, np.ones((X.shape[0], n_nodes), dtype=float)
        # flat: (batch, nodes * seq * feat)
        need = n_nodes * sl * fd
        if X.ndim == 2:
            if X.shape[1] < need:
                X = np.pad(X, ((0, 0), (0, need - X.shape[1])))
            X = X[:, :need].reshape(-1, n_nodes, sl, fd)
        mask = np.ones((X.shape[0], n_nodes), dtype=float)
        return X, mask

    def build_graph_from_history(self, history_returns: np.ndarray) -> np.ndarray:
        corr = shrinkage_correlation(history_returns, self.config.shrinkage)
        return adjacency_from_correlation(corr)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> GraphFIJEPAModel:
        node_mask = kwargs.get("node_mask")
        X_arr, default_mask = self._parse_X(X)
        if node_mask is not None:
            default_mask = np.asarray(node_mask, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, self.config.n_nodes)
        # history returns for graph: mean feature per node over time
        # history-only returns: average batch -> (time, nodes)
        rets = X_arr[:, :, :, 0]
        hist = rets.mean(axis=0).T
        self._frozen_adj = self.build_graph_from_history(hist)

        x_t = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(default_mask, dtype=torch.float32, device=self.device)
        adj = torch.as_tensor(self._frozen_adj, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        for _ in range(self.config.epochs):
            h = self.encoder(x_t, m_t)
            h = self.mp(h, adj)
            pred = self.head(h)
            loss = ((pred - y_t) ** 2 * m_t).sum() / m_t.sum().clamp(min=1.0)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self._frozen_adj is None:
            raise RuntimeError("Graph not built; call fit first.")
        use_shuffled = kwargs.get("shuffled_edges", False)
        adj_np = self._frozen_adj
        if use_shuffled:
            adj_np = shuffle_edges_control(adj_np, self._rng)
        node_mask = kwargs.get("node_mask")
        X_arr, default_mask = self._parse_X(X)
        if node_mask is not None:
            default_mask = np.asarray(node_mask, dtype=float)
        x_t = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(default_mask, dtype=torch.float32, device=self.device)
        adj = torch.as_tensor(adj_np, dtype=torch.float32, device=self.device)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x_t, m_t)
            h = self.mp(h, adj)
            pred = self.head(h)
            out = (pred * m_t).cpu().numpy()
        return np.asarray(out, dtype=float)

    def frozen_adjacency(self) -> np.ndarray:
        if self._frozen_adj is None:
            raise RuntimeError("No frozen graph")
        return self._frozen_adj.copy()

    def _serialize_state(self) -> Mapping[str, Any]:
        return {
            "config": self.config.to_dict(),
            "encoder": self.encoder.state_dict(),
            "mp": self.mp.state_dict(),
            "head": self.head.state_dict(),
            "frozen_adj": None if self._frozen_adj is None else self._frozen_adj.tolist(),
            "fitted": self._fitted,
        }

    def _deserialize_state(self, state: Mapping[str, Any]) -> None:
        self.config = M08Config.from_dict(state["config"])
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        h = self.config.hidden_dim
        self.encoder = _TemporalNodeEncoder(self.config.node_dim, h).to(self.device)
        self.mp = _GraphMessagePassing(h, self.config.message_passes).to(self.device)
        self.head = _FutureLatentHead(h, self.config.latent_dim).to(self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.mp.load_state_dict(state["mp"])
        self.head.load_state_dict(state["head"])
        fa = state.get("frozen_adj")
        self._frozen_adj = None if fa is None else np.asarray(fa, dtype=float)
        self._rng = np.random.default_rng(self.config.random_state)
        self._opt = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.mp.parameters())
            + list(self.head.parameters()),
            lr=self.config.lr,
        )
        self._fitted = bool(state.get("fitted", False))
