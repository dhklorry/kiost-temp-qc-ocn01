"""
Step 2 — 수온-염분(T-S) 회귀 잔차 + 해수 밀도(sigma-t) 피처
입력: work/feats.pkl   출력: work/Xts.npy, work/featnames_ts.json
주의: sigma_t()의 다항식 계수는 UNESCO EOS-80 해수 상태방정식의 물리 상수입니다.
      관측자료·재분석자료가 아니며 외부 데이터에 해당하지 않습니다.
"""
import numpy as np, pandas as pd, json, gc, os
WORK=os.environ.get('P1_WORK','work')

def sigma_t(T,S):
    T=np.asarray(T,float); S=np.asarray(S,float)
    rho_w=(999.842594+6.793952e-2*T-9.095290e-3*T**2+1.001685e-4*T**3
           -1.120083e-6*T**4+6.536332e-9*T**5)
    A=(8.24493e-1-4.0899e-3*T+7.6438e-5*T**2-8.2467e-7*T**3+5.3875e-9*T**4)
    B=(-5.72466e-3+1.0227e-4*T-1.6546e-6*T**2); C=4.8314e-4
    return rho_w+A*S+B*S**1.5+C*S**2-1000.0

def roll_reg_resid(y,x,w=2161,npass=2):
    """중심 롤링 OLS y~x, 이상값 2회 다운웨이팅 (수괴별 T-S 관계 국소 추정)"""
    sy=pd.Series(y); sx=pd.Series(x)
    wt=pd.Series(np.where(np.isfinite(y)&np.isfinite(x),1.0,0.0)); mp=max(50,w//10)
    for it in range(npass):
        Wy=sy.fillna(0)*wt; Wx=sx.fillna(0)*wt
        Sw=wt.rolling(w,center=True,min_periods=mp).sum()
        Sy=Wy.rolling(w,center=True,min_periods=mp).sum()
        Sx=Wx.rolling(w,center=True,min_periods=mp).sum()
        Sxx=(Wx*sx.fillna(0)).rolling(w,center=True,min_periods=mp).sum()
        Sxy=(Wx*sy.fillna(0)).rolling(w,center=True,min_periods=mp).sum()
        mx=Sx/Sw; my=Sy/Sw
        b=((Sxy/Sw-mx*my)/((Sxx/Sw-mx*mx)+1e-12)); a=my-b*mx
        r=sy-(a+b*sx)
        sc=(r.abs().rolling(w,center=True,min_periods=mp).median()*1.4826).clip(lower=1e-4)
        if it<npass-1:
            wt=pd.Series(np.where(np.isfinite(y)&np.isfinite(x),
                np.clip(1.0/(1.0+(np.abs(r.values)/(3*sc.values+1e-9))**2),0.0,1.0),0.0))
    return r.values, sc.values, b.values

def _r(v,w,fn,mp=None): return getattr(v.rolling(w,center=True,min_periods=mp or max(3,w//4)),fn)()

def build(featpath,outX,outNames):
    G=pd.read_pickle(featpath); G=G.loc[:,~G.columns.duplicated()]
    G=G.sort_values(['sid','time']).reset_index(drop=True)
    G['sig']=sigma_t(G.temp.values,G.psal.values)
    tsr=np.full(len(G),np.nan); tss=np.full(len(G),np.nan); tsb=np.full(len(G),np.nan)
    for sid,g in G.groupby('sid',sort=False):
        r,s,b=roll_reg_resid(g.temp.values,g.psal.values)
        tsr[g.index]=r; tss[g.index]=s; tsb[g.index]=b
    G['tsres']=tsr/np.maximum(tss,1e-4); G['tsslope']=tsb
    G['sy']=G.station+'|'+G.year.astype(str)
    info=G.groupby('sid').agg(sy=('sy','first'),dep=('depth','median'),lay=('layer','first'),st=('station','first'))
    dep_sl=G.groupby(['station','layer'])['depth'].median()
    info['dep']=info['dep'].fillna(pd.Series([dep_sl.get((r.st,r.lay),np.nan) for r in info.itertuples()],index=info.index)).fillna(info['lay']*5.0)
    piv=G.pivot_table(index='time',columns='sid',values='sig')
    cols={}
    for s in piv.columns:
        sy=info.loc[s,'sy']; d0=info.loc[s,'dep']
        sib=[c for c in piv.columns if info.loc[c,'sy']==sy and c!=s]
        up=sorted([c for c in sib if info.loc[c,'dep']<d0],key=lambda c:-info.loc[c,'dep'])
        dn=sorted([c for c in sib if info.loc[c,'dep']>d0],key=lambda c: info.loc[c,'dep'])
        a=piv[up[0]] if up else pd.Series(np.nan,index=piv.index)
        b=piv[dn[0]] if dn else pd.Series(np.nan,index=piv.index)
        cols[s]=pd.DataFrame({'dsig_up':piv[s]-a,'dsig_dn':b-piv[s],
                              'sigref':piv[sib].median(axis=1) if sib else np.nan})
    N=pd.concat(cols,names=['sid','time']).reset_index(); del cols,piv; gc.collect()
    G=G.merge(N,on=['sid','time'],how='left').sort_values(['sid','time']).reset_index(drop=True)
    G['sigres']=G['sig']-G['sigref']
    gb=G.groupby('sid',sort=False)
    for c in ['dsig_up','dsig_dn','sigres','sig']:
        G[c+'_dev']=gb[c].transform(lambda v: v-_r(v,2161,'median',100))
        G[c+'_z']=G[c+'_dev']/gb[c].transform(lambda v:(v-_r(v,2161,'median',100)).abs().rolling(2161,center=True,min_periods=100).median()*1.4826+1e-4)
    G['inv_up']=np.minimum(G['dsig_up'],0.0); G['inv_dn']=np.minimum(G['dsig_dn'],0.0)
    S={}
    for col in ['tsres','sigres_z','dsig_up_z','dsig_dn_z']:
        S[col]=G[col].values.astype(np.float32); gbc=G.groupby('sid',sort=False)[col]
        for h in [6,18,36,72,144,288]:
            wi,wo=2*h+1,6*h+1
            mi=gbc.transform(lambda v,w=wi:_r(v,w,'mean')); ma=gbc.transform(lambda v,w=wo:_r(v,w,'mean'))
            mo=(ma*wo-mi*wi)/(wo-wi); sdv=gbc.transform(lambda v,w=wo:_r(v,w,'std'))
            S[f'{col}_box{h}']=(mi-mo).values.astype(np.float32)
            S[f'{col}_boxn{h}']=((mi-mo)/(sdv+1e-6)).values.astype(np.float32)
            S[f'{col}_m{h}']=mi.values.astype(np.float32)
        for h in [18,72,144]:
            fwd=gbc.transform(lambda v,w=h: v.rolling(w,min_periods=w//2).mean().shift(-w))
            bwd=gbc.transform(lambda v,w=h: v.rolling(w,min_periods=w//2).mean())
            st=fwd-bwd; S[f'{col}_step{h}']=st.values.astype(np.float32)
            sa=pd.Series(np.abs(st.values)); g2=sa.groupby(G.sid.values,sort=False)
            for L in [288,576]:
                S[f'{col}_sB{h}_{L}']=g2.transform(lambda v,w=L: v.rolling(w,min_periods=1).max()).values.astype(np.float32)
                S[f'{col}_sF{h}_{L}']=g2.transform(lambda v,w=L: v[::-1].rolling(w,min_periods=1).max()[::-1]).values.astype(np.float32)
        n=len(G); t=pd.Series(np.arange(n,dtype=float))
        for w in [145,289,577]:
            mt=t.groupby(G.sid.values,sort=False).transform(lambda v,w=w:_r(v,w,'mean',w//2))
            mx=gbc.transform(lambda v,w=w:_r(v,w,'mean',w//2))
            mxt=(G[col]*t).groupby(G.sid.values,sort=False).transform(lambda v,w=w:_r(v,w,'mean',w//2))
            vt=(t*t).groupby(G.sid.values,sort=False).transform(lambda v,w=w:_r(v,w,'mean',w//2))
            S[f'{col}_ramp{w}']=(((mxt-mx*mt)/(vt-mt*mt+1e-9))*w).values.astype(np.float32)
        drz=gbc.transform(lambda v: v.diff().abs())
        for w in [19,73]:
            S[f'{col}_rough{w}']=np.log1p(drz.groupby(G.sid.values,sort=False).transform(lambda v,w=w:_r(v,w,'mean')).values).astype(np.float32)
    for c in ['tsslope','sig','sigres','dsig_up','dsig_dn','inv_up','inv_dn','sig_z','sigres_dev','dsig_up_dev','dsig_dn_dev']:
        S[c]=G[c].values.astype(np.float32)
    A=pd.DataFrame(S)
    np.save(outX,A.values.astype(np.float32)); json.dump(list(A.columns),open(outNames,'w'))
    print('step2 done',A.shape)

if __name__=='__main__':
    build(f'{WORK}/feats.pkl',f'{WORK}/Xts.npy',f'{WORK}/featnames_ts.json')
