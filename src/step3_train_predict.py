"""
Step 3 — 피처 선별(75개) → 모델 학습(5시드) → 후처리 → 제출 파일 생성
입력: work/feats.pkl, work/Xts.npy, data/sample_submission.csv
출력: submission.csv
모델: sklearn HistGradientBoostingClassifier (사전학습 가중치 없음, 매 실행 처음부터 학습)
"""
import numpy as np, pandas as pd, json, time, os
from sklearn.ensemble import HistGradientBoostingClassifier
DATA=os.environ.get('P1_DATA','data'); WORK=os.environ.get('P1_WORK','work')
OUT=os.environ.get('P1_OUT','submission.csv')
SEEDS=list(range(300,315))   # 15개 시드 평균 (분산 축소)
# 기지-연도 홀드아웃 CV 격자 탐색의 최적값 (tuning/derive_hyperparams.py 참조)
CFG=dict(max_leaf_nodes=20,min_samples_leaf=400,max_iter=250,learning_rate=0.05,
         max_features=0.5,l2_regularization=1.0,early_stopping=False)
# 후처리 상수: 기지-연도 홀드아웃 OOF 격자 탐색 결과 (tuning/derive_postproc.py 참조)
HI,LO,GAP,MINLEN,SPIKE_P=0.6,0.06,12,18,0.5

# 유형별로 균형 배분한 엄선 피처 75개 (독립 표본이 이상 세그먼트 263개뿐이라 과적합 방지가 최우선)
# 검증으로 선별한 75개 피처 (독립 표본이 이상 세그먼트 263개뿐 → 과적합 방지가 최우선.
# 404개 후보에서 기지-연도 홀드아웃 검증으로 축소; spike/flatline/noise/offset/drift 신호가
# 고르게 포함되도록 구성)
CUR=[
 'nd1n', 'nabs_d1', 'nabs_d1n', 'curv',
 'abs_curv', 'npd1', 'roughdev145', 'runlen',
 'log_runlen', 'zerofrac7', 'zerofrac19', 'zerofrac37',
 'roughratio7', 'roughratio37', 'roughp19', 'roughp73',
 'roughp145', 'roughratio289', 'roughdev19', 'rough37_box18',
 'rough37_box72', 'rough37_boxn72', 'zres_lay', 'zres_all_m289',
 'rough37_stpmaxB72_72', 'rough37_stpmaxF72_72', 'zres_lay_box6', 'zres_lay_box36',
 'zres_lay_box144', 'zres_lay_boxn36', 'zres_lay_boxn144', 'z_box36',
 'z_box144', 'z_boxn144', 'zres_all_box36', 'zres_all_box144',
 'zres_lay_stpmaxB18_576', 'z_stpmaxB72_576', 'zres_lay_stpmaxB72_72', 'zres_lay_stpmaxF72_72',
 'zres_lay_stpmaxF144_576', 'zres_lay_ramp73', 'zres_lay_ramp145', 'z_stpmaxF144_576',
 'z_ramp73', 'slope73', 'slope145', 'zres_lay_m145',
 'sc', 'depth', 'depth_dev', 'dev37',
 'dev145', 'zmean37', 'zmean145', 'depth',
 'depth', 'depth', 'strat_dn', 'strat_rng',
 'strat_up', 'rough37_ramp289', 'rough37_ramp577', 'doy_sin',
 'doy_cos', 'hr_cos', 'tsres_box72', 'tsres_box288',
 'tsres_boxn288', 'sigres_z_box288', 'tsres_ramp289', 'tsres',
 'sigres_z', 'dsig_up_z', 'dsig_dn_z',
]

def hyst(p,sid,hi,lo):
    """히스테리시스: 씨앗(>hi)을 포함하는 연결 구간을 lo까지 확장"""
    above=p>lo; n=len(p)
    newr=np.empty(n,bool); newr[0]=True
    newr[1:]=(above[1:]!=above[:-1])|(sid[1:]!=sid[:-1])
    gid=np.cumsum(newr)-1
    hi_any=np.zeros(gid[-1]+1,bool); np.maximum.at(hi_any,gid,p>hi)
    return (above&hi_any[gid]).astype(np.int8)

def _runs(pred,sid):
    n=len(pred); newr=np.empty(n,bool); newr[0]=True
    newr[1:]=(pred[1:]!=pred[:-1])|(sid[1:]!=sid[:-1])
    gid=np.cumsum(newr)-1; cnt=np.bincount(gid)
    starts=np.zeros(gid[-1]+1,dtype=np.int64); starts[gid[::-1]]=np.arange(n)[::-1]
    return gid,cnt,starts

def fillgap(pred,sid,maxgap):
    out=pred.copy(); gid,cnt,starts=_runs(pred,sid)
    for g in range(1,gid[-1]):
        i=starts[g]
        if pred[i]==0 and cnt[g]<=maxgap and sid[i]==sid[starts[g-1]] and sid[i]==sid[starts[g+1]] \
           and pred[starts[g-1]]==1 and pred[starts[g+1]]==1:
            out[i:i+cnt[g]]=1
    return out

def droptiny(pred,sid,p,minlen,spike_p):
    """최소 지속시간 미달 구간 제거 (단일점 spike는 확률이 높으면 유지)"""
    out=pred.copy(); gid,cnt,starts=_runs(pred,sid)
    for g in range(gid[-1]+1):
        i=starts[g]
        if pred[i]==1 and cnt[g]<minlen and not (cnt[g]==1 and p[i]>spike_p):
            out[i:i+cnt[g]]=0
    return out

def main():
    t0=time.time()
    G=pd.read_pickle(f'{WORK}/feats.pkl'); G=G.loc[:,~G.columns.duplicated()]
    Fts=json.load(open(f'{WORK}/featnames_ts.json')); Xts=np.load(f'{WORK}/Xts.npy')
    cols=[G[c].values.astype(np.float32) if c in G.columns else Xts[:,Fts.index(c)].astype(np.float32) for c in CUR]
    X=np.column_stack(cols); del Xts
    tri=(G.is_test==0).values
    y=G.label[tri].astype(int).values
    print('X',X.shape,'train',tri.sum(),'pos rate %.4f'%y.mean(),flush=True)
    pt=np.zeros((~tri).sum())
    for s in SEEDS:
        m=HistGradientBoostingClassifier(random_state=s,**CFG)
        m.fit(X[tri],y); pt+=m.predict_proba(X[~tri])[:,1]/len(SEEDS)
        print('seed',s,'%.0fs'%(time.time()-t0),flush=True)
    te=G[~tri][['sid','station','year','layer','time','runlen','rough37_box72','zres_lay_box144','z_ramp289']].copy()
    te['p']=pt; te=te.sort_values(['sid','time']).reset_index(drop=True)
    sid=te.sid.values; p=te.p.values
    pred=droptiny(fillgap(hyst(p,sid,HI,LO),sid,GAP),sid,p,MINLEN,SPIKE_P)
    te['label']=pred
    print('predicted positive rate %.4f (n=%d)'%(pred.mean(),pred.sum()),flush=True)
    # anomaly_type (선택 항목, 순위 미반영)
    n=len(te); newr=np.empty(n,bool); newr[0]=True
    newr[1:]=(pred[1:]!=pred[:-1])|(sid[1:]!=sid[:-1]); te['gid']=np.cumsum(newr)-1
    typ=pd.Series('',index=te.index)
    for g,d in te[te.label==1].groupby('gid'):
        if len(d)==1: t='spike'
        elif d.runlen.max()>=6: t='flatline'
        elif d.rough37_box72.mean()>0.35: t='noise'
        elif abs(d.z_ramp289.mean())>abs(d.zres_lay_box144.mean())*0.9: t='drift'
        else: t='offset'
        typ.loc[d.index]=t
    te['anomaly_type']=typ
    ss=pd.read_csv(f'{DATA}/sample_submission.csv')
    te['time_s']=te['time'].dt.strftime('%Y-%m-%dT%H:%M:%S+09:00')
    mm=te[['station','year','layer','time_s','label','anomaly_type']].rename(columns={'time_s':'time'})
    out=ss[['station','year','layer','time']].merge(mm,on=['station','year','layer','time'],how='left')
    assert out.label.notna().all(), '제출 키 정합 실패'
    out['label']=out['label'].astype(int); out['anomaly_type']=out['anomaly_type'].fillna('')
    out.to_csv(OUT,index=False)
    print('step3 done ->',OUT,out.shape,'total %.0fs'%(time.time()-t0))

if __name__=='__main__': main()
