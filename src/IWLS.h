#pragma once

#define MAX_N  80       // max number of experiments
#define MAX_K  4        // max number of regressors
#define MAX_M  100000   // max number of responses

extern "C" {
    void rlm_cpu(double *y, double *x, double *w, double *est, int *N, int *K, int *M, double *acc);
}
