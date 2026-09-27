"""피처 75개 선정의 근거 재현.
후보 풀 전체 / 무작위 75개 / 채택 75개 를 같은 홀드아웃 CV 로 비교한다.
독립 표본이 이상 세그먼트 263개뿐이므로 피처 수를 줄이는 것이 핵심이라는 점을 보인다.
train.csv 만 사용. 출력: tuning/out_features.json"""
import numpy as np, pandas as pd, json, time, os
from _common import *

if __name__=='__main__':
    G=pd.read_pickle(f'{WORK}/feats.pkl'); G=G.loc[:,~G.columns.duplicated()]
    Fts=json.load(open(f'{WORK}/featnames_ts.json'))
    DROP={'time','station','year','layer','temp','psal','label','anomaly_type','is_test','sid','obs','sy','bl',
          'zref_lay','zref_all','rgref_lay','rgref_all','rg19ref_lay','rg19ref_all','s1_series','month','layer_i'}
    pool=[c for c in G.columns if c not in DROP and G[c].dtype!=object]+Fts
    pool=[c for c in dict.fromkeys(pool)]
    print('후보 풀 %d개, 채택 %d개'%(len(pool),len(CUR)),flush=True)
    rng=np.random.default_rng(0)
    trials=[('채택 75개',CUR),('후보 풀 전체',pool)]
    for i in range(3):
        trials.append(('무작위 75개 #%d'%(i+1), list(rng.choice(pool,size=len(CUR),replace=False))))
    # 메모리 제약상 모든 시행을 동일한 행 부분표집으로 수행한다 (양성 전량 + 음성 50%,
    # 음성에 가중치 2 를 주어 확률 보정을 유지). 시행 간 비교 조건은 동일하다.
    out=[]; t0=time.time()
    for name,feats in trials:
        X,meta=load_train(feats)
        y=meta.label.astype(int).values
        rs=np.random.default_rng(1)
        keep=(y==1)|(rs.random(len(y))<0.5)
        w=np.where(y==1,1.0,2.0)[keep]
        Xs=X[keep]; ms=meta[keep].reset_index(drop=True); del X
        oof_s=holdout_oof(Xs,ms,CFG,[0],w=w); del Xs
        r=evaluate(ms,oof_s,HI,LO,GAP,MINLEN)
        meta=ms; oof=oof_s
        out.append(dict(name=name,n=len(feats),f1=float(r['f1']),rate=float(r['rate'])))
        print('  %-16s n=%3d  F1 %.5f  (%.0fs)'%(name,len(feats),r['f1'],time.time()-t0),flush=True)
        pass
    print('\n=== 결과 ===')
    for d in sorted(out,key=lambda d:-d['f1']): print('  F1 %.5f  %-16s (n=%d)'%(d['f1'],d['name'],d['n']))
    json.dump(out,open(os.path.join(os.path.dirname(__file__),'out_features.json'),'w'),indent=1)
