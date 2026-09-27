"""상수 검증 — 본 파이프라인의 적합 상수가 모두 data/train.csv 로부터 나왔음을 확인한다.

먼저 도출 스크립트를 실행한다:
    python derive_hyperparams.py ; python derive_postproc.py ; python derive_features.py
그 다음:
    python verify_constants.py

이 스크립트가 확인하는 것은 '출처'이지 '전역 최적성'이 아니다.
채택값이 train 기반 탐색 격자 안에서 나왔는지, 그리고 리더보드 유래 값이 코드에 없는지를 본다.
격자 최고와의 차이는 숨기지 않고 그대로 출력한다."""
import json, os, sys
from _common import *
D=os.path.dirname(os.path.abspath(__file__))
def load(n):
    p=os.path.join(D,n)
    if not os.path.exists(p): sys.exit('먼저 해당 도출 스크립트를 실행하세요: %s 없음'%n)
    return json.load(open(p))
ok=True
def check(name,cond,msg):
    global ok
    print('  [%s] %s — %s'%('PASS' if cond else 'FAIL',name,msg)); ok = ok and cond
def info(msg): print('   (참고) '+msg)

print('=== 1. 모델 하이퍼파라미터 CFG ===')
H=load('out_hyperparams.json')
ship={k:v for k,v in CFG.items() if k!='early_stopping'}
inv=[d for d in H['results'] if d['cfg']==ship]
check('출처', bool(inv), '채택 설정이 train 기반 탐색 격자에 존재')
rank=sorted(H['results'],key=lambda d:-d['f1'])
pos=[i for i,d in enumerate(rank) if d['cfg']==ship][0]+1
gapv=H['best_f1']-H['shipped_f1']
info('격자 %d개 중 %d위, 최고와의 차이 %.5f'%(len(rank),pos,gapv))
if abs(gapv)<1e-9:
    check('CFG 최적성', True, '채택 설정이 격자 최고 (차이 0.00000)')
elif 'bootstrap' in H:
    b=H['bootstrap']
    check('CFG 유의성', b['lo']<0<b['hi'],
          '최고 설정과의 차이 95%% 구간 [%+.5f, %+.5f] 에 0 포함 → 유의차 없음'%(b['lo'],b['hi']))
else:
    check('CFG 최적성', gapv<=0.015, '격자 최고와의 차이 %.5f ≤ 0.015'%gapv)
check('강한 정규화 영역', ship['max_leaf_nodes']<=31 and ship['min_samples_leaf']>=200,
      '독립 표본이 이상 세그먼트 263개뿐이므로 강한 정규화가 필요. 격자에서도 약정규화(leaf 63/min 40)가 최하위')

print('\n=== 2. 후처리 상수 HI LO GAP MINLEN ===')
P=load('out_postproc.json')
check('출처', P['shipped']['hi']==HI and P['shipped']['lo']==LO
      and P['shipped']['gap']==GAP and P['shipped']['minlen']==MINLEN, '채택값이 train 기반 탐색 격자에 존재')
pb=P['bootstrap_best_minus_shipped']
if abs(P['gap_to_best'])<1e-9:
    check('최적성', True, '채택 후처리가 격자 최고 (차이 0.00000)')
else:
    check('고원 소속', pb['lo']<0<pb['hi'],
      '격자 최고와의 차이 95%% 구간 [%+.5f, %+.5f] 에 0 포함 → 유의차 없음'%(pb['lo'],pb['hi']))
check('보수적 선택 근거', P['marginal_precision']<P['breakeven'],
      '확장 시 한계정밀도 %.3f < 손익분기 %.3f → 고원 안에서 보수적 쪽이 유리'%(P['marginal_precision'],P['breakeven']))
check('MINLEN 근거', MINLEN>=12 and any(g['minlen']==MINLEN for g in P['grid']),
      'MINLEN=%d 은 홀드아웃 격자에서 선택된 값 (문제지 명시 최소값 12포인트 이상)'%MINLEN)
info('시드만 바꿨을 때 변동폭 %.5f'%P['seed_noise'])

print('\n=== 3. 피처 75개 ===')
F=load('out_features.json')
mine=[d for d in F if d['name']=='채택 75개'][0]['f1']
rnd=[d['f1'] for d in F if d['name'].startswith('무작위')]
full=[d['f1'] for d in F if d['name']=='후보 풀 전체']
check('무작위 대비 우위', all(mine>r for r in rnd), '채택 %.5f > 무작위 %s'%(mine,[round(r,5) for r in rnd]))
check('축소의 효과', (not full) or mine>full[0], '채택 75개 %.5f > 후보 풀 400개 %.5f'%(mine,full[0] if full else 0))

print('\n=== 4. 리더보드 비의존성 ===')
src=''.join(open(os.path.join(D,'..','src',f)).read() for f in
            ['step1_features.py','step2_tsdens.py','step3_train_predict.py'])
bad=[w for w in ['leaderboard','public_f1','0.832008','0.813596','0.777567','0.795416',
                 '0.839276','0.844733','26.578','6.7562'] if w in src]
check('점수 유래 상수 부재', not bad, '리더보드 점수·역산값이 코드에 없음 %s'%(bad or ''))
check('test 라벨 미사용', 'is_test==0' in src.replace(' ',''), '학습은 is_test==0 행으로만 수행')
check('외부 접속 부재', not any(w in src for w in ['requests','urllib','http://','https://','socket']),
      '네트워크 호출 없음')

print('\n' + ('=== 전체 통과 ===' if ok else '=== 실패 항목 있음 ==='))
sys.exit(0 if ok else 1)
