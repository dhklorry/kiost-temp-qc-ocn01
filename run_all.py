"""P1 수온 이상탐지 — 단일 진입점. python run_all.py"""
import subprocess,sys,os,time
os.makedirs('work',exist_ok=True)
t0=time.time()
for s in ['src/step1_features.py','src/step2_tsdens.py','src/step3_train_predict.py']:
    print(f'\n===== {s} =====',flush=True)
    r=subprocess.run([sys.executable,s])
    if r.returncode!=0: sys.exit(f'FAILED: {s}')
print('\nALL DONE  총 %.1f분  -> submission.csv'%((time.time()-t0)/60))
