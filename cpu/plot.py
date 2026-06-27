import pandas as pd
import matplotlib.pyplot as plt

cpu_results = pd.read_csv("cpu_results.csv")

for N in sorted(cpu_results["N"].unique()):
    df = cpu_results[cpu_results["N"] == N].copy()
    labels = df["version"] + "_T" + df["threads"].astype(str) + "_BS" + df["BS"].astype(str)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, df["avg_time_s"])
    plt.xticks(rotation=90)
    plt.ylabel("Tempo médio (s)")
    plt.title(f"Tempo médio por versão — N={N}")
    plt.tight_layout()
    filename = f"avg_time_N{N}.png"
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()

for N in sorted(cpu_results["N"].unique()):
    df = cpu_results[cpu_results["N"] == N].copy()
    labels = df["version"] + "_T" + df["threads"].astype(str) + "_BS" + df["BS"].astype(str)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, df["GFLOPS"])
    plt.xticks(rotation=90)
    plt.ylabel("GFLOP/s")
    plt.title(f"Desempenho por versão — N={N}")
    plt.tight_layout()
    filename = f"gflops_N{N}.png"
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()
