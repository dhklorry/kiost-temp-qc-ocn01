"""튜닝 스크립트 공통부 — 모든 계산은 data/train.csv 에서만 이루어진다.
리더보드 점수, test.csv 의 정답, 그 밖의 외부 정보는 일절 사용하지 않는다."""
import numpy as np, pandas as pd, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
WORK=os.environ.get('P1_WORK', os.path.join(ROOT,'work'))

# 본 파이프라인이 사용하는 상수 (src/step3_train_predict.py 와 동일해야 함)
import importlib.util
_spec=importlib.util.spec_from_file_location('s3', os.path.join(os.path.dirname(__file__),'..','src','step3_train_predict.py'))
_s3=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s3)
CUR=_s3.CUR; CFG=_s3.CFG; SEEDS=_s3.SEEDS
HI,LO,GAP,MINLEN,SPIKE_P=_s3.HI,_s3.LO,_s3.GAP,_s3.MINLEN,_s3.SPIKE_P
hyst,fillgap,droptiny=_s3.hyst,_s3.fillgap,_s3.droptiny

HOLDOUT=['S-ORS|2024','S-ORS|2025','I-ORS|2025','G-ORS|2025']   # 기지-연도 단위 홀드아웃

def load_train(feats=None):
    """학습 구간의 피처 행렬 X 와 메타를 반환 (test 행은 사용하지 않는다).
    메모리 절약: feats.pkl 에서 필요한 열만 임시 memmap 에 옮기고 원본을 해제한 뒤 X 를 구성한다."""
    import gc, tempfile
    names=list(CUR if feats is None else feats)
    G=pd.read_pickle(f'{WORK}/feats.pkl'); G=G.loc[:,~G.columns.duplicated()]
    tri=(G.is_test==0).values; n=int(tri.sum())
    meta=G.loc[tri,['sid','station','year','layer','time','label']].reset_index(drop=True)
    ing=[(j,c) for j,c in enumerate(names) if c in G.columns]
    tmp=tempfile.NamedTemporaryFile(suffix='.dat',delete=False); tmp.close()
    mm=np.memmap(tmp.name,dtype=np.float32,mode='w+',shape=(n,max(1,len(ing))))
    for k,(j,c) in enumerate(ing):
        mm[:,k]=G[c].values[tri].astype(np.float32)
    mm.flush(); del G, mm; gc.collect()
    X=np.empty((n,len(names)),dtype=np.float32)
    mm=np.memmap(tmp.name,dtype=np.float32,mode='r',shape=(n,max(1,len(ing))))
    for k,(j,c) in enumerate(ing): X[:,j]=mm[:,k]
    del mm; gc.collect(); os.unlink(tmp.name)
    rest=[(j,c) for j,c in enumerate(names) if c not in set(c2 for _,c2 in ing)]
    if rest:
        Fts=json.load(open(f'{WORK}/featnames_ts.json'))
        Xts=np.load(f'{WORK}/Xts.npy',mmap_mode='r')
        for j,c in rest: X[:,j]=np.asarray(Xts[:,Fts.index(c)])[tri].astype(np.float32)
        del Xts; gc.collect()
    return X, meta

def f1(y,pr):
    tp=((pr==1)&(y==1)).sum(); fp=((pr==1)&(y==0)).sum(); fn=((pr==0)&(y==1)).sum()
    return 2*tp/max(1,2*tp+fp+fn)

def holdout_oof(X, meta, cfg, seeds, w=None):
    """기지-연도 홀드아웃 OOF 확률. 미지 연도 일반화를 재는 유일한 방법."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    sy=(meta.station+'|'+meta.year.astype(str)).values
    y=meta.label.astype(int).values
    oof=np.zeros(len(meta))
    for h in HOLDOUT:
        b=(sy==h); acc=np.zeros(b.sum())
        for s in seeds:
            m=HistGradientBoostingClassifier(random_state=s,**cfg)
            m.fit(X[~b],y[~b],sample_weight=None if w is None else w[~b])
            acc+=m.predict_proba(X[b])[:,1]/len(seeds)
        oof[b]=acc
        print('   holdout %-12s done'%h,flush=True)
    return oof

def evaluate(meta, oof, hi, lo, gap, minlen, spike_p=0.5, jan_jun=True):
    """테스트 모사 평가: 홀드아웃 OOF 를 1~6월 구간으로 제한해 채점.
    test.csv 가 2026-01~06 만 담고 있으므로 평가 구간을 맞춘다."""
    d=meta.assign(p=oof).sort_values(['sid','time'])
    y=d.label.astype(int).values; sid=d.sid.values; p=d.p.values
    m=(d.time.dt.month<=6).values if jan_jun else np.ones(len(d),bool)
    q=droptiny(fillgap(hyst(p,sid,hi,lo),sid,gap),sid,p,minlen,spike_p)
    tp=((q[m]==1)&(y[m]==1)).sum(); fp=((q[m]==1)&(y[m]==0)).sum(); fn=((q[m]==0)&(y[m]==1)).sum()
    return dict(f1=2*tp/max(1,2*tp+fp+fn), rate=q[m].mean(), tp=int(tp), fp=int(fp), fn=int(fn))
