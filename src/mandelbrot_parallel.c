/*
 * Mandelbrot
 *
 * Compilação:
 * gcc Serial_mandelbrot.c -lm -fopenmp -o mandelbrot
 *
 * 
 * Execução:
 * ./mandelbrot
 */

int MAXITER = 0; // MAX Iterations on mandelbrot
int N = 0; // Matrix Resolution

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

const double TOTAL_FLOPS_PER_ITERATION = 10;

/* ------------------------------------------------------------ */
/* Mandelbrot computation                                       */
/* ------------------------------------------------------------ */
void compute_mandelbrot(){
    register int i, j, k;
    register float complex z;
    register float complex kappa;
    float **x;

    
    /* Allocation */
    x = (float **)malloc(N * sizeof(float *));
    if(!x){
        fprintf(stderr, "Allocation failed.\n");
        exit(1);
    }

    for(i = 0; i < N; i++){
        x[i] = (float *)malloc(N * sizeof(float));
        if(!x[i]){
            fprintf(stderr, "Allocation failed.\n");
            exit(1);
        }
    }

    double start = omp_get_wtime();
    // Algorithm START:
    //===================================
    #pragma omp parallel for private(j,k,z,kappa) schedule(runtime)
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            kappa = 
                (4.0f * (i - N / 2)) / N +      // ---------------
                (4.0f * (j - N / 2)) / N * I;   // Total: 10 FLOPS
            z = kappa;
            k = 1;

            // CABSF -> sqrt(z_real^2 + z_imag^2) (3 FLOPS)
            while((cabsf(z) <= 2.0f) && (k++ < MAXITER)){
            
                // (z_real * z_real) - (z_imag * z_imag) + kappa_real
                // (2.0 * z_real * z_imag) + kappa_imag (7 FLOPS)
                z = z * z + kappa;
            } // 10 * K FLOPS 
            
            // logf -> É calculado de forma que levam MUITOS FLOPS (aproximações polinômiais)
            // A quantidade pode variar entre arquiteturas. 
            x[i][j] = logf((float)k) / logf((float)MAXITER); // 2 logf FLOPS
        }
    }   /*
        *   AI Aproximado: (10 + 10K + 1) / 4 (Funções SQRT e LOGF ignoradas na contagem)
        *       (1000 ITER) Melhor caso: K = 1, 21 / 4 
        *                       AI = 5.25
        *       (1000 ITER) Pior caso: K = 1000, 10011 / 4
        */
    //===================================
    // Algorithm END;

    double end = omp_get_wtime();
    double elapsed = end - start;
    

    uint64_t checksum = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            checksum ^= *(uint32_t*)&x[i][j];
        }
    }
    // N = 4096 ITER 1500
    // 406323905; // N = 2048 ITER = 1000
    // 387468970; // N = 2000 ITER = 1000
    uint64_t total_iterations = 406323905;

    //fprintf(stdout, "%ld\n", total_iterations);
    double total_flops = ((double)total_iterations * TOTAL_FLOPS_PER_ITERATION) + ((N*N) * 11);
    //fprintf(stdout, "%f\n", total_flops);
    //uint64_t bytes_moved = (uint64_t)((uint64_t)N * (uint64_t)N) * sizeof(float);
    //double aproxAI = total_flops / bytes_moved;
    double performance_gflops = (total_flops / elapsed) / 1e9; // (op / tempo) / gb
    
    //fprintf(stdout, "=========================================================\n");
    //fprintf(stdout, "Threads: %d\n", omp_get_thread_limit()); // Máximo de threads possível
    fprintf(stdout, "%f", elapsed);
    fprintf(stdout, ";%f", performance_gflops);
    //fprintf(stdout, "Bytes moved: %ld\n", bytes_moved);
    //fprintf(stdout, "Aprox AI: %f\n", aproxAI);
    //fprintf(stdout, "checksum: %ld\n", checksum);

    
    /* Cleanup */
    for(i = 0; i < N; i++){
        free(x[i]);
    }
    free(x);
}

/* ------------------------------------------------------------ */
/* Main */
/* ------------------------------------------------------------ */
int main(int argc, char *argv[]){
    N = atoi(argv[1]);
    MAXITER = atoi(argv[2]);

    compute_mandelbrot();

    return 0;
}
