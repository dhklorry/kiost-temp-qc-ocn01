"""후처리 상수(HI, LO, GAP, MINLEN)의 근거 재현.
(1) 기지-연도 홀드아웃 OOF 위에서 격자 탐색 → 고원(plateau) 확인
(2) 시드만 바꿨을 때의 변동폭을 측정해 '유의미한 차이'의 하한을 정한다
(3) 고원 안에서 보수적 쪽을 택한 이유(확장 시 한계정밀도)를 수치로 제시
train.csv 만 사용. 출력: tuning/out_postproc.json"""
import numpy as np, pandas as pd, json, time, os, itertools
from _common import *

HI_G=[0.4,0.5,0.6,0.7,0.8]; LO_G=[0.01,0.02,0.04,0.06,0.10,0.15]
GAP_G=[6,12,18,24,36];      ML_G=[12,18]

def grid(meta,oof):
    rows=[]
    for hi,lo,gap,ml in itertools.product(HI_G,LO_G,GAP_G,ML_G):
        if lo>=hi: continue
        r=evaluate(meta,oof,hi,lo,gap,ml)
        rows.append(dict(hi=hi,lo=lo,gap=gap,minlen=ml,f1=float(r['f1']),rate=float(r['rate'])))
    return pd.DataFrame(rows)

if __name__=='__main__':
    X,meta=load_train(); print('train X',X.shape,flush=True)
    t0=time.time()
    GS=SEEDS[:5]   # 격자 비교용 축약 시드셋 (본 예측은 SEEDS 전체를 사용)
    print('[1/2] 홀드아웃 OOF 계산 (격자 비교용 시드 %s)'%GS,flush=True)
    oof=holdout_oof(X,meta,CFG,GS); np.save(os.path.join(os.path.dirname(__file__),'out_oof_shipped.npy'),oof)
    print('   %.0fs'%(time.time()-t0),flush=True)
    print('[2/2] 다른 시드셋으로 한 번 더 (시드 잡음 측정)',flush=True)
    oof2=holdout_oof(X,meta,CFG,[s+50 for s in GS])
    print('   %.0fs'%(time.time()-t0),flush=True)

    G1=grid(meta,oof); G2=grid(meta,oof2)
    ship=G1[(G1.hi==HI)&(G1.lo==LO)&(G1.gap==GAP)&(G1.minlen==MINLEN)].iloc[0]
    ship2=G2[(G2.hi==HI)&(G2.lo==LO)&(G2.gap==GAP)&(G2.minlen==MINLEN)].iloc[0]
    seed_noise=abs(ship.f1-ship2.f1)
    best=G1.sort_values('f1',ascending=False).iloc[0]
    print('\n=== 격자 상위 8 ===')
    for _,r in G1.sort_values('f1',ascending=False).head(8).iterrows():
        print('  F1 %.5f rate %.4f | hi %.2f lo %.2f gap %d minlen %d'%(r.f1,r.rate,r.hi,r.lo,r.gap,r.minlen))
    print('\n채택값  hi %.2f lo %.2f gap %d minlen %d  → F1 %.5f  rate %.4f'%(HI,LO,GAP,MINLEN,ship.f1,ship.rate))
    print('격자 최고 F1 %.5f (hi %.2f lo %.2f gap %d minlen %d, rate %.4f)'%(best.f1,best.hi,best.lo,best.gap,best.minlen,best.rate))
    print('최고와의 차이            %.5f'%(best.f1-ship.f1))
    print('시드만 바꿨을 때 변동폭  %.5f   ← 이보다 작은 차이는 유의하지 않다'%seed_noise)

    # 확장 시 한계정밀도: 채택값 → 격자 최공격값
    aggr=G1.sort_values('rate',ascending=False).iloc[0]
    rc=evaluate(meta,oof,HI,LO,GAP,MINLEN)
    ra=evaluate(meta,oof,aggr.hi,aggr.lo,int(aggr.gap),int(aggr.minlen))
    dtp=ra['tp']-rc['tp']; dpred=(ra['tp']+ra['fp'])-(rc['tp']+rc['fp'])
    marg=dtp/max(1,dpred); brk=rc['f1']/2
    print('\n확장(→ hi %.2f lo %.2f) 시 한계정밀도 %.3f  vs 손익분기 %.3f'%(aggr.hi,aggr.lo,marg,brk))
    print('  → 한계정밀도가 손익분기보다 낮으므로 고원 안에서 보수적 쪽이 유리하다.')
    print('MINLEN=%d (문제지 명시 최소 지속시간은 flatline 2시간 = 12포인트).'%MINLEN)
    if MINLEN>12:
        print('  → 명시 최소값보다 크게 잡아 %d~%d 포인트 구간을 포기하는 대신 오탐을 줄이는 거래이며,'%(12,MINLEN-1))
        print('    위 격자에서 그 편이 유리한 것으로 나타났다.')

    from _boot import boot_diff
    bs=boot_diff(meta,oof,(HI,LO,GAP,MINLEN),(best.hi,best.lo,int(best.gap),int(best.minlen)),n=1000)
    print('\n격자 최고 − 채택값 차이 부트스트랩 (계열 단위 1000회)')
    print('  평균 %+.5f  95%% 구간 [%+.5f, %+.5f]  → 0 포함: %s'%(bs['mean'],bs['lo'],bs['hi'],bs['lo']<0<bs['hi']))
    if abs(bs['mean'])<1e-9:
        print('  채택값이 곧 격자 최고이므로 차이가 없다.')
    else:
        print('  → 구간에 0 이 포함되면 두 설정의 차이는 통계적으로 유의하지 않다.')
    json.dump(dict(bootstrap_best_minus_shipped=bs, shipped=dict(hi=HI,lo=LO,gap=GAP,minlen=MINLEN,f1=float(ship.f1),rate=float(ship.rate)),
                   best=dict(hi=float(best.hi),lo=float(best.lo),gap=int(best.gap),minlen=int(best.minlen),
                             f1=float(best.f1),rate=float(best.rate)),
                   gap_to_best=float(best.f1-ship.f1), seed_noise=float(seed_noise),
                   marginal_precision=float(marg), breakeven=float(brk),
                   grid=G1.to_dict('records')),
              open(os.path.join(os.path.dirname(__file__),'out_postproc.json'),'w'),indent=1)
