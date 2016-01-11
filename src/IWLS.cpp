#include <math.h>
#include <algorithm>
#include <vector>
#include <functional>
#include <cstring>
#include <limits>

#include <Rconfig.h>
#include <R.h>

#include "IWLS.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace IWLS {

    double  x[MAX_N][MAX_K];            // regressors
    double  y[MAX_M][MAX_N];            // response 
    bool   av[MAX_M][MAX_N];            // response is available 
    double  c[MAX_M][MAX_K];            // coefficients
    double cc[MAX_M][MAX_K];            // complete coefficients
    double  w[MAX_M][MAX_N];            // weights      
    int     o[MAX_M];                   // offset of noncomplete data
    double sg[MAX_M];                   // robust dispersion of response
    double  r[MAX_M][MAX_N];            // residue
    double pr[MAX_M][MAX_N];            // previous residue
    double ar[MAX_M][MAX_N];            // abs residue

    //utility for OLS
    double    a[MAX_K][MAX_N];          // x(T)
    double    b[MAX_M][MAX_N][MAX_K];   // xw
    double   xy[MAX_M][MAX_K];          // y*xw
    double sums[MAX_M][MAX_K][MAX_K];   // (xw*x(T))(-1)

    //utility Gauss-Jordan elimination
    double e[MAX_N][MAX_N];             //initial matrix

    int N,                              // number of experiments 
        K,                              // number of regressors
        M;                              // number of responses

    const double k = 1.345;             // Huber loss function param
    const double delta = 0.01;          // accuracy
    const double med = 0.6745;          // robust median coeff
    const int maxit = 20;               // max number of iterations


    void setPrevResidue()
    {
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < M; ++m)
            std::memcpy(pr[m], r[m], sizeof(double) * N);
    }

    //Gauss-Jordan elimination O(N^3)
    void  inversion(double A[MAX_K][MAX_K], int N)
    {
        double temp;

        for (int n1 = 0; n1 < N; ++n1) 
            for (int n2 = 0; n2 < N; ++n2)
                e[n1][n2] = 0.0;
        for (int n = 0; n < N; ++n)
            e[n][n] = 1.0;


        for (int k = 0; k < N; k++) {
            //diagonal element
            temp = A[k][k];

            for (int j = 0; j < N; j++) {
                A[k][j] /= temp;
                e[k][j] /= temp;
            }
        
            for (int i = k + 1; i < N; i++) {
                temp = A[i][k];
                for (int j = 0; j < N; j++) {
                    A[i][j] -= A[k][j] * temp;
                    e[i][j] -= e[k][j] * temp;
                }
            }
        }

        for (int k = N - 1; k > 0; k--) {
            for (int i = k - 1; i >= 0; i--) {
                temp = A[i][k];

                for (int j = 0; j < N; j++) {
                    A[i][j] -= A[k][j] * temp;
                    e[i][j] -= e[k][j] * temp;
                }
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = e[i][j];
            }
        }
    }

    void calcResidue(){
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < M; ++m){
            for (int n = 0; n < N; ++n){
                if (av[m][n]) {
                    r[m][n] = y[m][n];
                    for (int k = 0; k < K; ++k){
                        r[m][n] -= c[m][k] * x[n][k];
                    }
                } else {
                    r[m][n] = 0;
                }
            }
        }
    }

    void calcSigma(){
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {       
                ar[m][n] = fabs(r[m][n]);   
            }
        
            int avNum = 0;      // number of available responses
            for (int n = 0; n < N; ++n) {
                if (av[m][n]) {
                    ++avNum;
                }
            }
            std::sort(ar[m], ar[m] + N, std::greater<double>());
 
            if (avNum % 2 == 1) {
                sg[m] = ar[m][avNum / 2] / med;
            } else {
                sg[m] = (ar[m][avNum / 2 - 1] + ar[m][avNum / 2]) / (2 * med);
            }
        }
    }

    //psi.huber <- function(u, k = 1.345, deriv=0)
    //{
    //    if(!deriv) return(pmin(1, k / abs(u)))
    //    abs(u) <= k
    //}
    double huber(double val)
    {
        return std::min<double>(1, k / fabs(val));
    }

    //w <- psi(resid/scale)
    void calcWeight(){
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                if (av[m][n]) {
                    w[m][n] = huber(r[m][n] / sg[m]);
                } else {
                    w[m][n] = 0;
                }
            }
        }
    }

    //irls.delta <- function(old, new)
    //        sqrt(sum((old - new)^2)/max(1e-20, sum(old^2)))
    //...
    //if(!is.null(test.vec)) convi <- irls.delta(testpv, get(test.vec))
    //...
    //done <- (convi <= acc)
    //if(done) break
    int stopIWLS(){
        int offset = 0;         //current offset 
        double numerator = 0;   //sum((old - new)^2)
        double denominator = 0; //max(1e-20, sum(old^2))

        for(int m = 0; m < M; ++m) {
            numerator = 0;   
            denominator = 0; 

            for (int n = 0; n < N; ++n){
                numerator += (pr[m][n] - r[m][n]) * (pr[m][n] - r[m][n]);
                denominator += pr[m][n] * pr[m][n];
            }

            denominator = std::max<double>(1.0e-20, denominator);
            
            if(sqrt(numerator/denominator) < delta) { // terminate calculations for y[m]
                //copy coefficients using offset
                ++offset;
                std::memcpy(cc[m + o[m]],c[m], sizeof(double)*K);
            } else {
                //move data due to offset
                if (offset != 0) {                      
                    std::memcpy(pr[m - offset], r[m], sizeof(double) * N);
                    std::memcpy( r[m - offset], r[m], sizeof(double) * N);
                    std::memcpy( y[m - offset], y[m], sizeof(double) * N);  
                    std::memcpy(av[m - offset], av[m], sizeof(double) * N);
                    std::memcpy( w[m - offset], w[m], sizeof(double) * N);
                    sg[m - offset] = sg[m];
                    o[m - offset] = o[m] + offset;
                } else {
                    std::memcpy(pr[m], r[m], sizeof(double) * N);
                }
            }
        }
        
        M -= offset;
        if (M ==0) {
            return 1;
        } else {
            return 0;
        }
    }

    void OLS(){
        //a = x(T)
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {   
                a[k][n] = x[n][k];
            }
        }
        
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < M; ++m) {
            // b = x*w
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    b[m][n][k] = x[n][k] * w[m][n];
                }
            }

            //sums = xw*x(T)
            for (int k1 = 0; k1 < K; ++k1) {
                for (int k2 = 0; k2 < K; ++k2) {                
                    sums[m][k1][k2] = 0;
                    for (int n = 0; n < N; ++n) {
                        sums[m][k1][k2] += b[m][n][k1] * a[k2][n];
                    }
                }
            }
    
            //Xy = y*xw
            for (int k = 0; k < K; ++k) {
                xy[m][k] = 0;
                for (int n = 0; n < N; ++n) {
                    if (av[m][n]) {
                        xy[m][k] += y[m][n] * b[m][n][k];
                    }
                }
            }

            // (xw*x(T))(-1)
            inversion(sums[m], K);

            // (xw*x(T)))(-1)(y*xw)
            for (int i = 0; i < K; i++) {
                c[m][i] = 0;
                for (int j = 0; j < K; j++) {
                    c[m][i] += sums[m][i][j] * xy[m][j];
                }
            }
        }

        calcResidue();
        calcSigma();
    }

    extern "C" {
    void rlm_cpu(double *y, double *x, double *w, double *est, int *N, int *K, int *M, double *acc)
    {
        //init
        IWLS::N = *N;
        IWLS::K = *K;
        IWLS::M = *M;

        for (int n = 0; n < IWLS::N; ++n) {
            IWLS::w[0][n] = sqrt(w[n]);
        }

        for (int n = 0; n < IWLS::N; ++n) {
            for (int k = 0; k < IWLS::K; ++k) {
                IWLS::x[n][k] = x[k * IWLS::N + n] * IWLS::w[0][n];
            }
        }

        for (int m = 0; m < IWLS::M; ++m) {
            o[m] = 0;
        }

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < IWLS::M; ++m) {
            for (int n = 0; n < IWLS::N; ++n) {
                IWLS::y[m][n] = y[m * IWLS::N + n] * IWLS::w[0][n];
            }
        }

        //create flags of na, exclude only-zero responses
        std::vector<int> onlyZeroY;
        std::vector<int> singularFactor;
        int offset = 0;         //current offset 

        for (int m = 0; m < IWLS::M; ++m) {
            bool onlyZeroF = true;
            int cFactor = 0;    //counter of factors

            for (int n = 0; n < IWLS::N; ++n) {
                if (ISNA(IWLS::y[m][n])) {
                    IWLS::av[m][n] = false;
                    IWLS::w[m][n] = 0.0;
                }
                else {
                    IWLS::av[m][n] = true;
                    IWLS::w[m][n] = 1.0;
                    ++cFactor;

                    if (IWLS::y[m][n] != 0) {
                        onlyZeroF = false;
                    }
                }
            }

            // check for na, only-zero. move data
            if (onlyZeroF && cFactor) {
                ++offset;
                onlyZeroY.push_back(m);
            }
            else if (cFactor < 2) {
                ++offset;
                singularFactor.push_back(m);
            }
            else if (offset != 0) {
                std::memcpy(IWLS::pr[m - offset], IWLS::r[m], sizeof(double) * IWLS::N);
                std::memcpy(IWLS::r[m - offset],  IWLS::r[m], sizeof(double) * IWLS::N);
                std::memcpy(IWLS::y[m - offset],  IWLS::y[m], sizeof(double) * IWLS::N);
                std::memcpy(IWLS::w[m - offset],  IWLS::w[m], sizeof(double) * IWLS::N);
                std::memcpy(IWLS::av[m - offset], IWLS::av[m], sizeof(double) * IWLS::N);
                IWLS::sg[m - offset] = IWLS::sg[m];
                IWLS::o[m - offset] = IWLS::o[m] + offset;
            }
        }
        IWLS::M -= offset;

        // begin IWLS
        int iter = 0;

        OLS();
        setPrevResidue();

        do {
            calcWeight();
            OLS();

            if (++iter > maxit) break;
        } while (stopIWLS() != 1);
        // end IWLS

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int m = 0; m < *M; ++m)
            for (int k = 0; k < IWLS::K; ++k)
                est[m * IWLS::K + k] = cc[m][k];

        // set only-zero responses estimate to zero
        for (std::vector<int>::const_iterator it = onlyZeroY.begin(); it != onlyZeroY.end(); ++it) {
            for (int j = 0; j < IWLS::K; ++j) {
                est[IWLS::K * (*it) + j] = 0;
            }
        }

        // set singular factors estimate to nan
        for (std::vector<int>::const_iterator it = singularFactor.begin(); it != singularFactor.end(); ++it) {
            for (int j = 0; j < IWLS::K; ++j) {
                est[IWLS::K * (*it) + j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    } // extern "C"
}   // namespace IWLS
