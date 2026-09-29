"""Synthetic causal data and predeclared task adapters, version 1.

Origin t observes returns strictly before t. Targets are returns[t:t+H].
All synthetic asset labels are anonymous; these are not historical market data.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

DEFAULT_SEEDS=(2026092901,2026092902,2026092903,2026092904,2026092905)
WORLDS=('null','linear_signal','regime_shift','heavy_tails')
MODEL_TITLES={
'M01':'Regime-Conditioned Distributional Boosting',
'M02':'Factor-Residualized Cross-Sectional Ranking',
'M03':'Conditional Nonlinear Factor Model',
'M04':'Financial APEN', 'M05':'Q-APEN', 'M06':'Financial JEPA',
'M07':'Eigen-JEPA / Conditional Covariance', 'M08':'Dynamic Graph Financial JEPA',
'M09':'Regime-Robust Multi-Timescale Representation',
'M10':'Conditional Flow-Matching Scenarios',
'M11':'Liability-Conditioned Distributional MPC', 'M12':'LAF-GMJEPA Composed Stack'}
ABLATIONS={'M01':'no_regimes','M02':'raw_return_labels','M03':'legacy_single_snapshot',
'M04':'no_episodic_memory','M05':'linear_experts_only','M06':'no_predictive_loss',
'M07':'no_context','M08':'no_edges','M09':'fixed_group_weights',
'M10':'no_context','M11':'fully_invested','M12':'no_memory'}

@dataclass
class Dataset:
    world: str
    seed: int
    returns: np.ndarray
    normalized: np.ndarray
    scale: float
    train_origins: np.ndarray
    test_origins: np.ndarray
    cutoff: int=240
    horizon: int=5
    lookback: int=20
    @property
    def sha256(self):
        return hashlib.sha256(np.ascontiguousarray(self.returns).tobytes()).hexdigest()


def make_dataset(world:str, seed:int, n_steps:int=320, cutoff:int=240, n_test:int=60)->Dataset:
    if world not in WORLDS: raise ValueError(f'Unknown world {world}')
    if cutoff<80 or cutoff+n_test+5>n_steps: raise ValueError('Insufficient train/test history')
    rng=np.random.default_rng(seed); k=5; r=np.zeros((n_steps,k))
    for t in range(1,n_steps):
        phi={'null':0.,'linear_signal':.65,'regime_shift':.5,'heavy_tails':.2}[world]
        sigma=.01
        if world=='regime_shift':
            phi=.5 if t<cutoff else -.3
            sigma=.008 if t<cutoff else .022
        noise=rng.standard_t(4,k+1)/np.sqrt(2) if world=='heavy_tails' else rng.normal(size=k+1)
        innovation=sigma*(.5*noise[0]+np.sqrt(.75)*noise[1:])
        r[t]=phi*r[t-1]+innovation
    scale=float(np.sqrt(np.mean(r[1:cutoff]**2)))
    # At cutoff, all training target horizons have matured: max target end = cutoff-1.
    train=np.arange(20,cutoff-5+1,dtype=int)
    test=np.arange(cutoff,cutoff+n_test,dtype=int)
    assert train.max()+5-1<test.min()
    return Dataset(world,seed,r,r/scale,scale,train,test,cutoff)


def windows(r:np.ndarray, origins:np.ndarray, length:int)->np.ndarray:
    return np.stack([r[t-length:t] for t in origins])

def future_paths(r:np.ndarray, origins:np.ndarray, horizon:int)->np.ndarray:
    return np.stack([r[t:t+horizon] for t in origins])

def point_features(r:np.ndarray, origins:np.ndarray)->np.ndarray:
    rows=[]
    for t in origins:
        h=r[t-20:t]
        rows.append([r[t-1,0],h[-5:,0].mean(),h[:,0].std(ddof=1),r[t-1].mean()])
    return np.asarray(rows)

def panel_features(r:np.ndarray, origins:np.ndarray):
    features=[]; betas=[]
    for t in origins:
        h=r[t-20:t]; market=h.mean(axis=1)
        centered=h-h.mean(axis=0); m=market-market.mean()
        beta=(centered*m[:,None]).sum(axis=0)/max(float(m@m),1e-10)
        x=np.column_stack([h[-1],h[-5:].mean(axis=0),h.std(axis=0,ddof=1),np.full(h.shape[1],market[-1]),beta])
        features.append(x); betas.append(beta)
    return np.stack(features),np.stack(betas)

def covariance_targets(paths):
    x=paths-paths.mean(axis=1,keepdims=True)
    return np.einsum('nhi,nhj->nij',x,x)/max(paths.shape[1]-1,1)

def energy_losses(samples, truth):
    """Unbiased ensemble energy-score estimate with distinct-pair second term."""
    a=np.asarray(samples); y=np.asarray(truth)
    if a.ndim!=3 or y.shape!=(a.shape[0],a.shape[2]): raise ValueError('Energy-score shape mismatch')
    first=np.linalg.norm(a-y[:,None,:],axis=2).mean(axis=1)
    if a.shape[1]<2: return first
    pair=np.linalg.norm(a[:,:,None,:]-a[:,None,:,:],axis=3).sum(axis=(1,2))/(a.shape[1]*(a.shape[1]-1))
    return first-.5*pair
