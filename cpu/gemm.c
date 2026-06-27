#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

static void fill_matrix(int N, float *M) {
    for (int i = 0; i < N*N; i++) {
        M[i] = rand_float();
    }
}

static void zero_matrix(int N, float *M) {
    memset(M, 0, sizeof(float) * N * N);
}

static double max_abs_error(int N, const float *A, const float *B) {
    double err = 0.0;
    for (int i = 0; i < N*N; i++) {
        double e = fabs((double)A[i] - (double)B[i]);
        if (e > err) err = e;
    }
    return err;
}

void gemm_naive(int N, const float *A, const float *B, float *C) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void transpose(int N, const float *B, float *BT) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[j*N + i] = B[i*N + j];
        }
    }
}

void gemm_transposed(int N, const float *A, const float *BT, float *C) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * BT[j*N + k];
            }
            C[i*N + j] = sum;
        }
    }
}

void gemm_blocked(int N, int BS, const float *A, const float *B, float *C) {

    for (int ii = 0; ii < N; ii += BS) {
        int i_max = (ii + BS < N) ? ii + BS : N;
        for (int kk = 0; kk < N; kk += BS) {
            int k_max = (kk + BS < N) ? kk + BS : N;
            for (int jj = 0; jj < N; jj += BS) {
                int j_max = (jj + BS < N) ? jj + BS : N;
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        float a = A[i*N + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i*N + j] += a * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_openmp(int N, const float *A, const float *B, float *C) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a = A[i*N + k];
            for (int j = 0; j < N; j++) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}

void gemm_blocked_openmp(int N, int BS, const float *A, const float *B, float *C) {

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BS) {
        int i_max = (ii + BS < N) ? ii + BS : N;
        for (int kk = 0; kk < N; kk += BS) {
            int k_max = (kk + BS < N) ? kk + BS : N;
            for (int jj = 0; jj < N; jj += BS) {
                int j_max = (jj + BS < N) ? jj + BS : N;
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        float a = A[i*N + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i*N + j] += a * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

static void usage(const char *prog) {
    printf("Usage: %s <version> <N> <repeats> [BS]\n", prog);
    printf("Versions: naive, transposed, blocked, openmp, blocked_openmp\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    const char *version = argv[1];
    int N = atoi(argv[2]);
    int repeats = atoi(argv[3]);
    int BS = (argc >= 5) ? atoi(argv[4]) : 32;

    srand(0);

    float *A, *B, *BT, *C, *Cref;
    posix_memalign((void**)&A, ALIGNMENT, sizeof(float)*N*N);
    posix_memalign((void**)&B, ALIGNMENT, sizeof(float)*N*N);
    posix_memalign((void**)&BT, ALIGNMENT, sizeof(float)*N*N);
    posix_memalign((void**)&C, ALIGNMENT, sizeof(float)*N*N);
    posix_memalign((void**)&Cref, ALIGNMENT, sizeof(float)*N*N);

    fill_matrix(N, A);
    fill_matrix(N, B);
    zero_matrix(N, C);
    zero_matrix(N, Cref);

    // Referência para validação.
    // Para N grande, isto pode demorar. Ainda assim, é importante para corretude.
    gemm_naive(N, A, B, Cref);

    if (strcmp(version, "transposed") == 0) {
        transpose(N, B, BT);
    }

    double best = 1e30;
    double total = 0.0;

    for (int r = 0; r < repeats; r++) {
        zero_matrix(N, C);
        double t0 = now_seconds();

        if (strcmp(version, "naive") == 0) {
            gemm_naive(N, A, B, C);
        } else if (strcmp(version, "transposed") == 0) {
            gemm_transposed(N, A, BT, C);
        } else if (strcmp(version, "blocked") == 0) {
            gemm_blocked(N, BS, A, B, C);
        } else if (strcmp(version, "openmp") == 0) {
            gemm_openmp(N, A, B, C);
        } else if (strcmp(version, "blocked_openmp") == 0) {
            gemm_blocked_openmp(N, BS, A, B, C);
        } else {
            usage(argv[0]);
            return 1;
        }

        double t1 = now_seconds();
        double elapsed = t1 - t0;
        if (elapsed < best) best = elapsed;
        total += elapsed;
    }

    double avg = total / repeats;
    double gflops = (2.0 * N * N * N) / (avg * 1e9);
    double err = max_abs_error(N, C, Cref);

    printf("version,N,BS,threads,repeats,avg_time_s,best_time_s,GFLOPS,max_abs_error\n");
    printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.8f\n",
           version, N, BS, omp_get_max_threads(), repeats, avg, best, gflops, err);

    free(A); free(B); free(BT); free(C); free(Cref);
    return 0;
}
