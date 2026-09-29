"""M07: conditional SPD covariance head with explicit forward-target support.

The historical API remains available for reconstruction diagnostics, but only
``future_covariances`` supplies forecasting targets. No target is inferred from
an unlabelled future slice, and past reconstruction is explicitly labelled.
"""
from __future__ import annotations
from typing import Any, ClassVar, Mapping, Optional
import numpy as np
from wharton_lab.contracts.base import ModelCapabilities, ModelMetadata
from wharton_lab.models.base import BaseModel
from wharton_lab.models.m07.config import M07Config
from wharton_lab.models.m07.metrics import chordal_distance, flag_small_eigengaps, projector_from_eigenvectors
from wharton_lab.models.m07.psd import ConditionalPSDCovHead
from wharton_lab.models.torch_utils import pick_device, require_torch, set_deterministic_seed

torch = require_torch()
nn = torch.nn

class _ReturnEncoder(nn.Module):
    def __init__(self, window: int, n_assets: int, hidden: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(window*n_assets, hidden), nn.ReLU(), nn.Linear(hidden, latent))
    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))

class EigenJEPAModel(BaseModel):
    MODEL_ID: ClassVar[str] = "M07"
    metadata = ModelMetadata(model_id="M07", title="Eigen-JEPA", version="0.2.0",
        description="Conditional SPD covariance regression; future targets must be explicit. Not a new JEPA pretraining objective.",
        inputs=("return_windows",), outputs=("covariance", "eigenspaces"),
        training_constraints=("explicit_matured_future_covariance",), metrics=("frobenius",), tags=("covariance",))

    def __init__(self, config: M07Config | None = None):
        self.config = config or M07Config()
        set_deterministic_seed(self.config.random_state)
        self.device = pick_device(self.config.device)
        c = self.config
        self.encoders = nn.ModuleDict({str(w): _ReturnEncoder(w,c.n_assets,c.hidden_dim,c.latent_dim) for w in c.windows}).to(self.device)
        self.cov_heads = nn.ModuleDict({str(w): ConditionalPSDCovHead(c.n_assets,c.latent_dim) for w in c.windows}).to(self.device)
        self._opt = torch.optim.Adam(list(self.encoders.parameters())+list(self.cov_heads.parameters()), lr=c.lr)
        self.scales_ = {str(w): 1.0 for w in c.windows}
        self._last_covariances = {}
        self._last_eigvals = {}; self._last_projectors = {}; self._last_eigengap_flags = {}
        self.target_mode_ = "not_fitted"
        self._fitted = False

    def capabilities(self):
        return ModelCapabilities(supports_factors=True)
    def config_dict(self):
        return self.config.to_dict()

    def _panel_from_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim not in (2,3) or len(X)==0 or not np.isfinite(X).all():
            raise ValueError("Finite nonempty 2D/3D context is required")
        out = {}; k = self.config.n_assets
        for w in self.config.windows:
            if X.ndim==3 and X.shape[2]==k and X.shape[1]>=w:
                out[w] = X[:, -w:, :]
            else:
                flat = X.reshape(len(X), -1)
                need = w*k
                if flat.shape[1]<need:
                    if getattr(self, "target_mode_", "") == "explicit_forward_covariance":
                        raise ValueError("Forward covariance contexts cannot be zero padded")
                    flat = np.pad(flat, ((0,0),(need-flat.shape[1],0)))
                out[w] = flat[:, -need:].reshape(-1,w,k)
        return out

    @staticmethod
    def _empirical_cov(window):
        centered = window-window.mean(axis=0,keepdims=True)
        return centered.T@centered/max(len(window)-1,1)

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        targets_arg = kwargs.get("future_covariances")
        self.target_mode_ = "explicit_forward_covariance" if targets_arg is not None else "past_reconstruction_not_forecasting"
        panels = self._panel_from_X(X)
        tensors = {}; targets = {}
        n = len(X); k = self.config.n_assets
        weights = np.ones(n) if sample_weight is None else np.asarray(sample_weight,dtype=float)
        if weights.shape!=(n,) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
            raise ValueError("Invalid sample weights")
        wt = torch.as_tensor(weights/weights.sum(), dtype=torch.float32,device=self.device)
        for w, arr in panels.items():
            cov = np.asarray(targets_arg[w],dtype=float) if targets_arg is not None else np.array([self._empirical_cov(a) for a in arr])
            if cov.shape!=(n,k,k) or not np.isfinite(cov).all():
                raise ValueError("Targets must be (n,assets,assets) finite covariances")
            if not np.allclose(cov,cov.swapaxes(-1,-2),atol=1e-8) or np.min(np.linalg.eigvalsh(cov)) < -1e-8:
                raise ValueError("Targets must be symmetric positive semidefinite")
            scale = max(float(np.trace(cov,axis1=1,axis2=2).mean()/k),1e-10)
            self.scales_[str(w)] = scale
            tensors[w] = torch.as_tensor(arr/np.sqrt(scale),dtype=torch.float32,device=self.device)
            targets[w] = torch.as_tensor(cov/scale,dtype=torch.float32,device=self.device)
        self.loss_history_ = []
        for _ in range(self.config.epochs):
            self.encoders.train(); self.cov_heads.train()
            loss = torch.zeros((),device=self.device)
            for w in panels:
                z = self.encoders[str(w)](tensors[w])
                cov = self.cov_heads[str(w)](z)
                loss = loss + (((cov-targets[w])**2).mean(dim=(1,2))*wt).sum()/len(panels)
            self._opt.zero_grad(); loss.backward(); self._opt.step()
            self.loss_history_.append(float(loss.detach().cpu()))
        self._fitted=True
        self._update_eigen_cache(panels)
        return self

    def predict_covariances(self, X, window):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        arr = self._panel_from_X(X)[window]
        scale = self.scales_[str(window)]
        self.encoders.eval(); self.cov_heads.eval()
        with torch.no_grad():
            t=torch.as_tensor(arr/np.sqrt(scale),dtype=torch.float32,device=self.device)
            cov=self.cov_heads[str(window)](self.encoders[str(window)](t))
        return cov.cpu().numpy().astype(float)*scale

    def _update_eigen_cache(self, panels):
        for w, arr in panels.items():
            # Cache is an aggregate diagnostic; batched forecasts use predict_covariances.
            scale=self.scales_[str(w)]
            self.encoders.eval(); self.cov_heads.eval()
            with torch.no_grad():
                t=torch.as_tensor(arr/np.sqrt(scale),dtype=torch.float32,device=self.device)
                cov=self.cov_heads[str(w)](self.encoders[str(w)](t)).cpu().numpy().mean(axis=0)*scale
            self._last_covariances[str(w)]=cov
            evals,evecs=np.linalg.eigh(cov)
            self._last_eigvals[str(w)]=evals
            self._last_projectors[str(w)]=projector_from_eigenvectors(evecs[:,-min(3,len(evals)):])
            self._last_eigengap_flags[str(w)]=flag_small_eigengaps(evals,self.config.eigengap_threshold)

    def predict(self,X,**kwargs):
        if not self._fitted: raise RuntimeError("Model not fitted")
        self._update_eigen_cache(self._panel_from_X(X))
        return np.concatenate([self._last_eigvals[str(w)][-3:] for w in self.config.windows])
    def predict_covariance(self,window):
        return self._last_covariances[str(window)].copy()
    def last_projector(self,window):
        return self._last_projectors[str(window)].copy()
    def chordal_to_empirical(self,X,window):
        arr=self._panel_from_X(X)[window]
        cov=np.mean([self._empirical_cov(a) for a in arr],axis=0)
        _,vec=np.linalg.eigh(cov)
        return chordal_distance(self.last_projector(window),projector_from_eigenvectors(vec[:,-min(3,len(vec)):]))
    def _serialize_state(self):
        return {"config":self.config.to_dict(),"encoders":self.encoders.state_dict(),"cov_heads":self.cov_heads.state_dict(),
            "scales":self.scales_,"target_mode":self.target_mode_,"covariances":self._last_covariances,
            "eigvals":self._last_eigvals,"projectors":self._last_projectors,"flags":self._last_eigengap_flags,"fitted":self._fitted,
            "format_version":2}
    def _deserialize_state(self,state):
        if state.get("format_version")!=2:
            raise ValueError("Legacy unconditional M07 checkpoint requires the original source snapshot")
        self.__init__(M07Config.from_dict(state["config"]))
        self.encoders.load_state_dict(state["encoders"]); self.cov_heads.load_state_dict(state["cov_heads"])
        self.scales_=state["scales"]; self.target_mode_=state["target_mode"]
        self._last_covariances=state["covariances"]; self._last_eigvals=state["eigvals"]
        self._last_projectors=state["projectors"]; self._last_eigengap_flags=state["flags"]
        self._fitted=state["fitted"]
