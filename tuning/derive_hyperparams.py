"""모델 하이퍼파라미터(CFG)의 근거 재현.
정규화 강도를 바꿔가며 기지-연도 홀드아웃 CV 로 비교한다. train.csv 만 사용.
출력: tuning/out_hyperparams.json"""
import numpy as np, json, time, os
from _common import *

GRID=[
 dict(max_leaf_nodes=15,min_samples_leaf=400,max_iter=200,learning_rate=0.06,max_features=0.6,l2_regularization=1.0,early_stopping=False),
 dict(max_leaf_nodes=20,min_samples_leaf=400,max_iter=250,learning_rate=0.05,max_features=0.5,l2_regularization=1.0,early_stopping=False),
 dict(max_leaf_nodes=15,min_samples_leaf=800,max_iter=150,learning_rate=0.06,max_features=0.7,l2_regularization=1.0,early_stopping=False),
 dict(max_leaf_nodes=31,min_samples_leaf=200,max_iter=300,learning_rate=0.06,max_features=0.6,l2_regularization=1.0,early_stopping=False),
 dict(max_leaf_nodes=63,min_samples_leaf=40, max_iter=500,learning_rate=0.06,max_features=0.6,l2_regularization=1.0,early_stopping=False),
 dict(max_leaf_nodes=10,min_samples_leaf=400,max_iter=300,learning_rate=0.07,max_features=0.6,l2_regularization=1.0,early_stopping=False),
]
if __name__=='__main__':
    X,meta=load_train(); print('train X',X.shape,flush=True)
    out=[]; t0=time.time()
    for i,cfg in enumerate(GRID):
        print(' [%d/%d] %s'%(i+1,len(GRID),{k:v for k,v in cfg.items() if k!='early_stopping'}),flush=True)
        oof=holdout_oof(X,meta,cfg,[0,1])
        np.save(os.path.join(os.path.dirname(__file__),'out_oof_cfg%d.npy'%i),oof)
        r=evaluate(meta,oof,HI,LO,GAP,MINLEN)
        out.append(dict(idx=i,cfg={k:v for k,v in cfg.items() if k!='early_stopping'},**{k:float(v) if k in('f1','rate') else v for k,v in r.items()}))
        print('     F1 %.5f  rate %.4f   (%.0fs)'%(r['f1'],r['rate'],time.time()-t0),flush=True)
    out.sort(key=lambda d:-d['f1'])
    print('\n=== 순위 ===')
    for d in out: print('  F1 %.5f  %s'%(d['f1'],d['cfg']))
    ship={k:v for k,v in CFG.items() if k!='early_stopping'}
    best=out[0]['f1']; mine=[d for d in out if d['cfg']==ship][0]['f1']
    print('\n패키지 채택 설정 %s'%ship)
    print('  F1 %.5f  |  격자 최고 %.5f  |  차이 %.5f'%(mine,best,best-mine))
    # 계열 단위 블록 부트스트랩: 최고 설정과 채택 설정의 차이가 유의한가?
    from _boot import boot_diff_two
    bi=out[0]['idx']; si=[d['idx'] for d in out if d['cfg']==ship][0]
    oof_s=np.load(os.path.join(os.path.dirname(__file__),'out_oof_cfg%d.npy'%si))
    oof_b=np.load(os.path.join(os.path.dirname(__file__),'out_oof_cfg%d.npy'%bi))
    # 동일 후처리에서 두 확률열을 비교하기 위해 각각 평가 후 차이를 부트스트랩
    bs=boot_diff_two(meta,oof_s,oof_b,(HI,LO,GAP,MINLEN))
    print('\n최고 설정 − 채택 설정 차이 부트스트랩 (계열 단위)')
    print('  평균 %+.5f  95%% 구간 [%+.5f, %+.5f]  → 0 포함: %s'%(bs['mean'],bs['lo'],bs['hi'],bs['lo']<0<bs['hi']))
    json.dump(dict(results=out,shipped=ship,shipped_f1=mine,best_f1=best,bootstrap=bs),
              open(os.path.join(os.path.dirname(__file__),'out_hyperparams.json'),'w'),indent=1)
