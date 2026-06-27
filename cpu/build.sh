#! /usr/bin/env bash

gcc gemm.c -O3 -march=native -fopenmp -fopt-info-vec-optimized -o gemm 2> vec_report.txt
