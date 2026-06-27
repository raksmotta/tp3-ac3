import subprocess
import pandas as pd
from io import StringIO

experiments = []
Ns = [128, 256, 512]
repeats = 5

for N in Ns:
    experiments.append(("naive", N, None, None))
    experiments.append(("transposed", N, None, None))
    for BS in [16, 32, 64]:
        experiments.append(("blocked", N, BS, None))
    for th in [1, 2, 4, 8]:
        experiments.append(("openmp", N, None, th))
        experiments.append(("blocked_openmp", N, 32, th))

rows = []
for version, N, BS, th in experiments:
    cmd = ["./gemm", version, str(N), str(repeats)]
    if BS is not None:
        cmd.append(str(BS))
    env = None
    if th is not None:
        env = {"OMP_NUM_THREADS": str(th)}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        continue
    lines = result.stdout.strip().splitlines()
    if len(lines) >= 2:
        df = pd.read_csv(StringIO("\n".join(lines[-2:])))
        rows.append(df)

cpu_results = pd.concat(rows, ignore_index=True)
cpu_results.to_csv("cpu_results.csv", index=False)
