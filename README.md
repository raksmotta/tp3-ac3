# Trabalho Prático 3 — Otimização Paralela de GEMM

Multiplicação de matrizes densas (`C = A × B`) partindo de uma versão ingênua e
evoluindo com localidade de memória, blocagem/tiling, vetorização SIMD, OpenMP e
CUDA. Disciplina de Arquitetura de Computadores.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `trabalho_pratico_3.ipynb` | Notebook original (enunciado, com os `TODO`). |
| `trabalho_pratico_3_resolvido.ipynb` | Notebook com todas as funções implementadas. |
| `trabalho_pratico_3_RODADO.ipynb` | Notebook executado no Colab, com os resultados. |
| `template/relatorio.tex` | Relatório final em LaTeX (template SBC2023). |
| `template/figs/` | Gráficos usados no relatório. |

## Como rodar (Google Colab)

1. Abra `trabalho_pratico_3_resolvido.ipynb` no Colab.
2. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente → GPU**.
3. Execute as células na ordem. Os resultados são salvos em
   `cpu_results.csv` e `gpu_results.csv`.

## Como compilar o relatório (Overleaf)

1. Faça upload de toda a pasta `template/`.
2. Defina `relatorio.tex` como documento principal.
3. Use o compilador **XeLaTeX** (a classe usa `fontspec`).
