#pragma once

#define MAX_N  80       // max number of experiments
#define MAX_K  4        // max number of regressors
#define MAX_M  100000   // max number of responses


#ifdef _WIN32
    //  Windows
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__linux__)
    //  Linux
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#elif defined(__APPLE__)
    //  OS X
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#endif

extern "C" {
    EXPORT void rlm_cpu(double *y, double *x, double *w, double *est, int *N, int *K, int *M, double *acc);
}

