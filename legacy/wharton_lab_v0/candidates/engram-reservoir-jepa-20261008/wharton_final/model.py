"""Trainable JEPA + contractive reservoir + matured engram correction.

This is development amendment A-20261008, not a reproduction of frozen M01–M12.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy, json
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from .data import matured_indices

@dataclass(frozen=True)
class Config:
    context: int = 128
    horizon: int = 20
    reporting_delay_seconds: int = 0
    latent: int = 16
    reservoir: int = 32
    spectral_radius: float = .8
    leak: float = .3
    epochs: int = 20
    learning_rate: float = .001
    ema: float = .99
    memory_slots: int = 256
    neighbors: int = 5
    memory_strength: float = .25
    head: str = 'xgboost'
    seed: int = 11
    def __post_init__(self):
        if self.head not in ('xgboost','ridge') or min(self.context,self.latent,self.reservoir,self.epochs,self.memory_slots,self.neighbors)<1 or self.horizon<1 or self.reporting_delay_seconds<0 or not 0<self.spectral_radius<1 or not 0<self.leak<=1 or not 0<=self.ema<1 or not 0<=self.memory_strength<=1 or self.learning_rate<=0:
            raise ValueError('invalid model config')
        if self.seed == 997: raise ValueError('spent lockbox seed prohibited')

class Encoder(nn.Module):
    def __init__(self,assets,dim):
        super().__init__();self.gru=nn.GRU(assets,dim,batch_first=True);self.projection=nn.Linear(dim,dim)
    def forward(self,x):
        _,h=self.gru(x);return self.projection(h[-1])

class FinalModel:
    def __init__(self, config=Config()): self.config=config;self.fitted=False
    def _check(self):
        if not self.fitted: raise RuntimeError('model not fitted')
    def _latent(self,x):
        with torch.no_grad():
            z=self.encoder(torch.tensor((x-self.mean)/self.scale,dtype=torch.float32)).numpy()
        # Each context starts from zero: inference is batch/order independent.
        states=np.zeros((len(x),self.config.reservoir))
        for t in range(x.shape[1]):
            obs=(x[:,t]-self.mean)/self.scale
            states=(1-self.config.leak)*states+self.config.leak*np.tanh(obs@self.win+states@self.wres)
        return np.concatenate([z,states],axis=1)
    def fit(self,s,as_of):
        self.fitted=False  # Failed refits must not expose partially replaced parameters.
        if not np.isfinite(as_of):raise ValueError('nonfinite fit clock')
        c=self.config
        if s.context.ndim!=3 or s.future.ndim!=3 or s.context.shape[1]!=c.context or s.future.shape[1]!=c.horizon or s.context.shape[-1]!=s.future.shape[-1] or s.labels.shape!=(len(s.context),s.context.shape[-1]):
            raise ValueError('sample shape/horizon differs from model contract')
        idx=matured_indices(s,as_of)
        if len(idx)<12: raise ValueError('at least 12 matured samples required')
        if np.any(np.diff(s.decisions[idx])<=0): raise ValueError('training samples must be chronological')
        split=max(8,int(.8*len(idx)));train=idx[:split];cal=idx[split:]
        # Purge training labels that mature after calibration starts.
        train=train[s.maturity[train]<s.decisions[cal[0]]]
        if len(train)<4: raise ValueError('insufficient purged fit/calibration history')
        x=s.context[train];future=s.future[train];y=s.labels[train]
        if any(not np.isfinite(a).all() for a in (x,future,y,s.context[cal],s.labels[cal])):
            raise ValueError('nonfinite eligible training/calibration data')
        self.assets=x.shape[-1];self.as_of=float(as_of)
        self.mean=x.reshape(-1,self.assets).mean(0);self.scale=x.reshape(-1,self.assets).std(0).clip(1e-8)
        torch.manual_seed(c.seed);torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.encoder=Encoder(self.assets,c.latent)
        target=copy.deepcopy(self.encoder).eval()
        for p in target.parameters(): p.requires_grad_(False)
        predictor=nn.Sequential(nn.Linear(c.latent,c.latent),nn.Tanh(),nn.Linear(c.latent,c.latent))
        aux=nn.Linear(c.latent,self.assets)
        opt=torch.optim.Adam(list(self.encoder.parameters())+list(predictor.parameters())+list(aux.parameters()),lr=c.learning_rate)
        tx=torch.tensor((x-self.mean)/self.scale,dtype=torch.float32)
        tf=torch.tensor((future-self.mean)/self.scale,dtype=torch.float32)
        ty=torch.tensor(y,dtype=torch.float32)
        self.losses=[]
        for _ in range(c.epochs):
            opt.zero_grad();z=self.encoder(tx)
            with torch.no_grad(): target_z=target(tf)
            latent_loss=(predictor(z)-target_z).square().mean()
            variance=torch.relu(1-torch.sqrt(z.var(0,unbiased=False)+1e-4)).mean()
            centered=z-z.mean(0);cov=centered.T@centered/max(1,len(z)-1)
            offdiag=cov-torch.diag(torch.diag(cov))
            loss=latent_loss+.1*(aux(z)-ty).square().mean()+.1*variance+.01*offdiag.square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite JEPA loss')
            loss.backward();nn.utils.clip_grad_norm_(self.encoder.parameters(),1);opt.step()
            with torch.no_grad():
                for tp,p in zip(target.parameters(),self.encoder.parameters()):tp.mul_(c.ema).add_(p,alpha=1-c.ema)
            self.losses.append(float(loss.detach()))
        self.encoder.eval()
        rng=np.random.default_rng(c.seed)
        self.win=rng.normal(scale=1/np.sqrt(self.assets),size=(self.assets,c.reservoir))
        w=rng.normal(size=(c.reservoir,c.reservoir))
        # Operator norm bound guarantees contraction, stronger than radius alone.
        self.wres=w*(c.spectral_radius/np.linalg.norm(w,2))
        z=self._latent(x);self.zmean=z.mean(0);self.zscale=z.std(0).clip(1e-8)
        z=(z-self.zmean)/self.zscale
        self.heads=[]
        for a in range(self.assets):
            if c.head=='xgboost':
                try: from xgboost import XGBRegressor
                except ImportError as e: raise RuntimeError('XGBoost required for full model; install xgboost, or explicitly select ridge ablation') from e
                head=XGBRegressor(n_estimators=64,max_depth=3,learning_rate=.05,subsample=1,colsample_bytree=1,random_state=c.seed,n_jobs=1,objective='reg:squarederror')
            else: head=Ridge(alpha=1)
            head.fit(z,y[:,a]);self.heads.append(head)
        calz=(self._latent(s.context[cal])-self.zmean)/self.zscale
        raw=np.column_stack([h.predict(calz) for h in self.heads])
        self.keys=calz[-c.memory_slots:];self.residuals=(s.labels[cal]-raw)[-c.memory_slots:]
        self.memory_maturity=s.maturity[cal][-c.memory_slots:].copy()
        self.train_indices=train.copy();self.calibration_indices=cal.copy()
        self.train_rows=s.rows[train].tolist();self.calibration_rows=s.rows[cal].tolist()
        self.fitted=True
        return self
    def predict(self,x,as_of,*,use_memory=True):
        self._check();x=np.asarray(x,float)
        if not np.isfinite(as_of) or as_of<self.as_of or x.ndim!=3 or x.shape[1:]!=(self.config.context,self.assets) or not np.isfinite(x).all(): raise ValueError('invalid inference context/clock')
        z=(self._latent(x)-self.zmean)/self.zscale
        prediction=np.column_stack([h.predict(z) for h in self.heads])
        usable=self.memory_maturity<as_of
        keys=self.keys[usable];residuals=self.residuals[usable]
        if len(keys) and use_memory:
            norm=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-12)
            query=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            sims=query@norm.T
            for i,row in enumerate(sims):
                top=np.argsort(row,kind='stable')[-self.config.neighbors:]
                weights=np.maximum(row[top],0)
                if weights.sum()>0: prediction[i]+=self.config.memory_strength*(weights@residuals[top])/weights.sum()
        return prediction
    def save(self,path):
        """Local trusted checkpoint; never load third-party pickle checkpoints."""
        import pickle
        self._check()
        with Path(path).open('wb') as f:pickle.dump(self,f,protocol=5)
    @staticmethod
    def load(path):
        import pickle
        with Path(path).open('rb') as f: model=pickle.load(f)
        if not isinstance(model,FinalModel):raise ValueError('wrong checkpoint type')
        return model
