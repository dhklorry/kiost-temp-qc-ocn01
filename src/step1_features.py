"""
Step 1 — 기본 시계열 피처 생성
입력: data/train.csv, data/test.csv
출력: work/feats.pkl
외부 데이터/네트워크 미사용. numpy/pandas만 사용.
"""
import pandas as pd, numpy as np, os
DATA=os.environ.get('P1_DATA','data'); WORK=os.environ.get('P1_WORK','work')
GRID='10min'

def load_all():
    tr=pd.read_csv(f'{DATA}/train.csv',parse_dates=['time'])
    te=pd.read_csv(f'{DATA}/test.csv',parse_dates=['time'])
    tr['is_test']=0; te['is_test']=1; te['label']=np.nan; te['anomaly_type']=np.nan
    df=pd.concat([tr,te],ignore_index=True)
    df['sid']=df['station']+'|'+df['year'].astype(str)+'|'+df['layer'].astype(str)
    return df

def regrid(df):
    out=[]
    for sid,g in df.groupby('sid',sort=False):
        g=g.sort_values('time').set_index('time')
        idx=pd.date_range(g.index[0],g.index[-1],freq=GRID)
        g2=g.reindex(idx); g2['sid']=sid
        g2['obs']=(~g2['temp'].isna()).astype(np.int8)&g2['station'].notna().astype(np.int8)
        for c in ['station','year','layer','is_test']: g2[c]=g[c].iloc[0]
        g2.index.name='time'; out.append(g2.reset_index())
    return pd.concat(out,ignore_index=True)

def daily_baseline(g,col,days=15):
    """일 중앙값 -> 15일 롤링 중앙값 기준선 + 롤링 MAD 스케일 (기지·층·계절별 자동 표준화)"""
    s=g.set_index('time')[col]
    med=s.resample('1D').median()
    mad=(s-s.resample('1D').transform('median')).abs().resample('1D').median()
    mr=med.rolling(days,center=True,min_periods=3).median()
    ar=mad.rolling(days,center=True,min_periods=3).median()
    bl=mr.reindex(s.index.union(mr.index)).interpolate(limit_direction='both').reindex(s.index)
    sc=ar.reindex(s.index.union(ar.index)).interpolate(limit_direction='both').reindex(s.index)
    return bl.values, sc.values

def runlen_equal(x):
    """연속 동일값 런 길이 (flatline 탐지)"""
    n=len(x); eq=np.zeros(n,dtype=bool)
    eq[1:]=(x[1:]==x[:-1])&~np.isnan(x[1:])
    gid=np.cumsum(~eq)-1
    return np.bincount(gid)[gid].astype(np.float32)

def series_feats(g):
    f={}
    x=g['temp'].to_numpy(float); p=g['psal'].to_numpy(float); dep=g['depth'].to_numpy(float)
    n=len(x); sx=pd.Series(x); sp=pd.Series(p); sd=pd.Series(dep)
    bl,sc=daily_baseline(g,'temp'); sc=np.maximum(sc,1e-3)
    f['bl']=bl; f['sc']=sc; z=(x-bl)/sc; f['z']=z
    blp,scp=daily_baseline(g,'psal'); scp=np.maximum(scp,1e-3); f['zp']=(p-blp)/scp
    d1=sx.diff(); d1n=-sx.diff(-1)
    s1=max(np.nanmedian(np.abs(d1.to_numpy())),1e-4); f['s1_series']=np.full(n,s1)
    f['nd1']=(d1/s1).to_numpy(); f['nd1n']=(d1n/s1).to_numpy()
    f['nabs_d1']=np.abs(f['nd1']); f['nabs_d1n']=np.abs(f['nd1n'])
    f['min_absd']=np.minimum(f['nabs_d1'],f['nabs_d1n'])
    curv=(sx.shift(1)+sx.shift(-1)-2*sx)/s1
    f['curv']=curv.to_numpy(); f['abs_curv']=np.abs(f['curv'])
    dp1=sp.diff(); sp1=max(np.nanmedian(np.abs(dp1.to_numpy())),1e-5)
    f['npd1']=np.abs((dp1/sp1).to_numpy()); f['ndep1']=np.abs(sd.diff().to_numpy())
    ad1=d1.abs()
    for w in [7,19,37,73,145,289]:
        r=ad1.rolling(w,center=True,min_periods=max(3,w//4)).mean()
        f[f'rough{w}']=np.log1p((r/s1).to_numpy())
        rp=dp1.abs().rolling(w,center=True,min_periods=max(3,w//4)).mean()
        f[f'roughp{w}']=np.log1p((rp/sp1).to_numpy())
        f[f'roughratio{w}']=f[f'rough{w}']-f[f'roughp{w}']   # 염분엔 이상 미주입 → noise 직격
    for w in [19,73,145]:
        rr=pd.Series(f[f'rough{w}'])
        f[f'roughdev{w}']=(rr-rr.rolling(2161,center=True,min_periods=100).median()).to_numpy()
    rl=runlen_equal(x); f['runlen']=rl; f['log_runlen']=np.log1p(rl)
    zero=(d1.abs()<1e-12).astype(float)
    for w in [7,19,37,73]:
        f[f'zerofrac{w}']=zero.rolling(w,center=True,min_periods=3).mean().to_numpy()
    for w in [7,19,37,73,145,289]:
        m=sx.rolling(w,center=True,min_periods=max(3,w//4)).median()
        f[f'dev{w}']=((sx-m)/sc).to_numpy()
    zs=pd.Series(z)
    for w in [7,37,145,289,577]:
        f[f'zmean{w}']=zs.rolling(w,center=True,min_periods=max(3,w//4)).mean().to_numpy()
    t=np.arange(n,dtype=float)
    for w in [73,145,289]:
        mx=zs.rolling(w,center=True,min_periods=w//2).mean()
        mt=pd.Series(t).rolling(w,center=True,min_periods=w//2).mean()
        cov=(zs*pd.Series(t)).rolling(w,center=True,min_periods=w//2).mean()-mx*mt
        var=(pd.Series(t*t)).rolling(w,center=True,min_periods=w//2).mean()-mt*mt
        f[f'slope{w}']=(cov/var*w).to_numpy()
    f['depth']=dep
    f['depth_dev']=(sd-sd.rolling(145,center=True,min_periods=20).median()).to_numpy()
    return f

def _roll(v,w,fn='mean',mp=None):
    return getattr(v.rolling(w,center=True,min_periods=mp or max(3,w//4)),fn)()

def contrast_feats(G,cols):
    """박스카/계단/램프 정합필터 — offset·drift 탐지의 핵심"""
    out={}
    for col in cols:
        gb=G.groupby('sid',sort=False)[col]
        for h in [6,18,36,72,144,288]:
            wi,wo=2*h+1,6*h+1
            mi=gb.transform(lambda v,w=wi:_roll(v,w)); ma=gb.transform(lambda v,w=wo:_roll(v,w))
            mo=(ma*wo-mi*wi)/(wo-wi); sdv=gb.transform(lambda v,w=wo:_roll(v,w,'std'))
            box=mi-mo
            out[f'{col}_box{h}']=box.values.astype(np.float32)
            out[f'{col}_boxn{h}']=(box/(sdv+1e-6)).values.astype(np.float32)
        for h in [18,72,144]:
            fwd=gb.transform(lambda v,w=h: v.rolling(w,min_periods=w//2).mean().shift(-w))
            bwd=gb.transform(lambda v,w=h: v.rolling(w,min_periods=w//2).mean())
            st=fwd-bwd; out[f'{col}_step{h}']=st.values.astype(np.float32)
            sa=pd.Series(np.abs(st.values),index=G.index); g2=sa.groupby(G['sid'].values,sort=False)
            for L in [72,288,576]:
                out[f'{col}_stpmaxB{h}_{L}']=g2.transform(lambda v,w=L: v.rolling(w,min_periods=1).max()).values.astype(np.float32)
                out[f'{col}_stpmaxF{h}_{L}']=g2.transform(lambda v,w=L: v[::-1].rolling(w,min_periods=1).max()[::-1]).values.astype(np.float32)
        for w in [73,145,289,577]:
            n=len(G); t=pd.Series(np.arange(n,dtype=float),index=G.index)
            mt=t.groupby(G['sid'].values,sort=False).transform(lambda v,w=w:_roll(v,w,'mean',w//2))
            mx=gb.transform(lambda v,w=w:_roll(v,w,'mean',w//2))
            mxt=(G[col]*t).groupby(G['sid'].values,sort=False).transform(lambda v,w=w:_roll(v,w,'mean',w//2))
            vt=(t*t).groupby(G['sid'].values,sort=False).transform(lambda v,w=w:_roll(v,w,'mean',w//2))
            out[f'{col}_ramp{w}']=(((mxt-mx*mt)/(vt-mt*mt+1e-9))*w).values.astype(np.float32)
    return pd.DataFrame(out,index=G.index)

def build():
    G=regrid(load_all()).sort_values(['sid','time']).reset_index(drop=True)
    parts=[pd.DataFrame(series_feats(g),index=g.index) for _,g in G.groupby('sid',sort=False)]
    G=pd.concat([G,pd.concat(parts).sort_index()],axis=1)
    G['sy']=G['station']+'|'+G['year'].astype(str)
    # 층간 / 기지간 기준계열 (leave-one-out 중앙값)
    for col,tag in [('z','z'),('rough73','rg'),('rough19','rg19')]:
        piv=G.pivot_table(index='time',columns='sid',values=col)
        sid2sy=G.groupby('sid')['sy'].first()
        groups={}
        for s,v in sid2sy.items(): groups.setdefault(v,[]).append(s)
        yearmap={s:s.split('|')[1] for s in piv.columns}
        rl=pd.DataFrame(index=piv.index,columns=piv.columns,dtype=np.float32)
        ra=pd.DataFrame(index=piv.index,columns=piv.columns,dtype=np.float32)
        for s in piv.columns:
            same=[c for c in groups[sid2sy[s]] if c!=s]
            if same: rl[s]=piv[same].median(axis=1)
            oth=[c for c in piv.columns if yearmap[c]==yearmap[s] and sid2sy[c]!=sid2sy[s]]
            if oth: ra[s]=piv[oth].median(axis=1)
        for nm,R in [('lay',rl),('all',ra)]:
            st=R.stack(future_stack=True).rename(f'{tag}ref_{nm}').reset_index()
            st.columns=['time','sid',f'{tag}ref_{nm}']
            G=G.merge(st,on=['time','sid'],how='left')
    for tag,col in [('z','z'),('rg','rough73'),('rg19','rough19')]:
        for nm in ['lay','all']: G[f'{tag}res_{nm}']=G[col]-G[f'{tag}ref_{nm}']
    G=G.sort_values(['sid','time'])
    for base in ['zres_lay','zres_all']:
        s=G.groupby('sid',sort=False)[base]
        for w in [7,37,145,289]:
            G[f'{base}_m{w}']=s.transform(lambda v,w=w: v.rolling(w,center=True,min_periods=max(3,w//4)).mean())
        G[f'{base}_sl']=s.transform(lambda v: v.rolling(145,center=True,min_periods=40).mean().diff(72))
    for base in ['rgres_lay','rgres_all','rg19res_lay']:
        G[f'{base}_m37']=G.groupby('sid',sort=False)[base].transform(lambda v: v.rolling(37,center=True,min_periods=8).mean())
    CF=contrast_feats(G,['z','zres_lay','zres_all','rough37'])
    G=pd.concat([G.reset_index(drop=True),CF.reset_index(drop=True)],axis=1)
    t=G['time']; doy=t.dt.dayofyear.values; hr=t.dt.hour.values+t.dt.minute.values/60
    G['doy_sin']=np.sin(2*np.pi*doy/365.25); G['doy_cos']=np.cos(2*np.pi*doy/365.25)
    G['hr_sin']=np.sin(2*np.pi*hr/24); G['hr_cos']=np.cos(2*np.pi*hr/24)
    G['month']=t.dt.month; G['layer_i']=G['layer'].astype(int)
    G['station_i']=G['station'].map({'G-ORS':0,'I-ORS':1,'S-ORS':2}).astype(int)
    # 성층 지표: 인접 층과의 온도차 + 계절 편차
    piv=G.pivot_table(index='time',columns='sid',values='temp')
    info=G.groupby('sid').agg(sy=('sy','first'),layer=('layer','first'))
    add={}
    for s in piv.columns:
        sy=info.loc[s,'sy']; lay=info.loc[s,'layer']
        sibs=[c for c in piv.columns if info.loc[c,'sy']==sy and c!=s]
        up=[c for c in sibs if info.loc[c,'layer']==lay-1]; dn=[c for c in sibs if info.loc[c,'layer']==lay+1]
        a=piv[up[0]] if up else pd.Series(np.nan,index=piv.index)
        b=piv[dn[0]] if dn else pd.Series(np.nan,index=piv.index)
        add[s]=pd.DataFrame({'strat_up':piv[s]-a,'strat_dn':b-piv[s],
                             'strat_rng':piv[sibs].max(axis=1)-piv[sibs].min(axis=1) if sibs else np.nan})
    S=pd.concat(add,names=['sid','time']).reset_index()
    G=G[G['obs']==1].reset_index(drop=True)
    G=G.merge(S,on=['sid','time'],how='left').sort_values(['sid','time']).reset_index(drop=True)
    gb=G.groupby('sid',sort=False)
    for c in ['strat_up','strat_dn','strat_rng']:
        G[c+'_dev']=gb[c].transform(lambda v: v-v.rolling(2161,center=True,min_periods=100).median())
    return G

if __name__=='__main__':
    os.makedirs(WORK,exist_ok=True)
    G=build(); G.to_pickle(f'{WORK}/feats.pkl'); print('step1 done',G.shape)
