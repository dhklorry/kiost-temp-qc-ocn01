"""계열 단위 블록 부트스트랩 — 두 설정의 F1 차이가 통계적으로 유의한지 판정.
평가 구간의 독립 표본은 행이 아니라 '계열'과 '이상 세그먼트'이므로 계열 단위로 재표집한다."""
import numpy as np, pandas as pd
from _common import *

def boot_diff(meta, oof, A, B, n=2000, seed=0):
    d=meta.assign(p=oof).sort_values(['sid','time']).reset_index(drop=True)
    y=d.label.astype(int).values; sid=d.sid.values; p=d.p.values
    m=(d.time.dt.month<=6).values
    def counts(cfg):
        q=droptiny(fillgap(hyst(p,sid,cfg[0],cfg[1]),sid,cfg[2]),sid,p,cfg[3],0.5)
        return q
    qa=counts(A); qb=counts(B)
    sids=pd.unique(sid); idx={s:np.where((sid==s)&m)[0] for s in sids}
    rng=np.random.default_rng(seed); diffs=[]
    def f1c(ii,q):
        tp=((q[ii]==1)&(y[ii]==1)).sum(); fp=((q[ii]==1)&(y[ii]==0)).sum(); fn=((q[ii]==0)&(y[ii]==1)).sum()
        return 2*tp/max(1,2*tp+fp+fn)
    for _ in range(n):
        pick=rng.choice(len(sids),size=len(sids),replace=True)
        ii=np.concatenate([idx[sids[k]] for k in pick])
        diffs.append(f1c(ii,qb)-f1c(ii,qa))
    diffs=np.array(diffs)
    return dict(mean=float(diffs.mean()), lo=float(np.quantile(diffs,0.025)),
                hi=float(np.quantile(diffs,0.975)), p_gt0=float((diffs>0).mean()))


def boot_diff_two(meta, oofA, oofB, cfg, n=1000, seed=0):
    """같은 후처리 설정에서 서로 다른 두 확률열(A=채택, B=비교)의 F1 차이 부트스트랩"""
    d=meta.sort_values(['sid','time']).reset_index(drop=True)
    order=meta.sort_values(['sid','time']).index.values
    y=d.label.astype(int).values; sid=d.sid.values; m=(d.time.dt.month<=6).values
    def pred(oof):
        p=oof[order]
        return droptiny(fillgap(hyst(p,sid,cfg[0],cfg[1]),sid,cfg[2]),sid,p,cfg[3],0.5)
    qa=pred(oofA); qb=pred(oofB)
    sids=pd.unique(sid); idx={s:np.where((sid==s)&m)[0] for s in sids}
    rng=np.random.default_rng(seed); diffs=[]
    def f1c(ii,q):
        tp=((q[ii]==1)&(y[ii]==1)).sum(); fp=((q[ii]==1)&(y[ii]==0)).sum(); fn=((q[ii]==0)&(y[ii]==1)).sum()
        return 2*tp/max(1,2*tp+fp+fn)
    for _ in range(n):
        pick=rng.choice(len(sids),size=len(sids),replace=True)
        ii=np.concatenate([idx[sids[k]] for k in pick])
        diffs.append(f1c(ii,qb)-f1c(ii,qa))
    diffs=np.array(diffs)
    return dict(mean=float(diffs.mean()), lo=float(np.quantile(diffs,0.025)),
                hi=float(np.quantile(diffs,0.975)), p_gt0=float((diffs>0).mean()))
