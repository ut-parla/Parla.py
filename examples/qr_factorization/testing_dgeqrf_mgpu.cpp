/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_zgeqrf_mgpu.cpp, normal z -> d, Thu Oct  8 23:05:42 2020
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t    H2D_time, gpu_time, D2H_time;
    double *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
    magmaDouble_ptr d_lA[ MagmaMaxGPUs ];
    magma_int_t M, N, n2, lda, lwork, info, min_mn, nb;
    magma_int_t ldda, n_local, ngpu;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
 
    int status = 0;

    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_create( dev, &queues[dev] );
    }
    
    printf("%% ngpu %lld\n", (long long) opts.ngpu );
    printf("%%==============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            nb     = magma_get_dgeqrf_nb( M, N );
            
            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, magma_ceildiv(N,nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %lld GPUs\n", (long long) ngpu );
            }
            
            // query for workspace size
            lwork = -1;
            lapackf77_dgeqrf( &M, &N, unused, &M, unused, tmp, &lwork, &info );
            lwork = (magma_int_t) MAGMA_D_REAL( tmp[0] );
            lwork = max( lwork, N*nb );
            
            // Allocate host memory for the matrix
            TESTING_CHECK( magma_dmalloc_cpu( &tau,    min_mn ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_work, lwork  ));
            
            TESTING_CHECK( magma_dmalloc_pinned( &h_R,    n2     ));
            
            // Allocate device memory
            for( int dev = 0; dev < ngpu; dev++ ) {
                n_local = ((N/nb)/ngpu)*nb;
                if (dev < (N/nb) % ngpu)
                    n_local += nb;
                else if (dev == (N/nb) % ngpu)
                    n_local += N % nb;
                magma_setdevice( dev );
                TESTING_CHECK( magma_dmalloc( &d_lA[dev], ldda*n_local ));
            }
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, M, N, h_A, lda );
            lapackf77_dlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            //printf("M = %lld   N = %lld\n", (long long) M, (long long) N );

            H2D_time = magma_wtime();
            magma_dsetmatrix_1D_col_bcyclic( ngpu, M, N, nb, h_R, lda, d_lA, ldda, queues );
            H2D_time = magma_wtime() - H2D_time;
            printf("\"%f\",", H2D_time); // H2D

            gpu_time = magma_wtime();
            magma_dgeqrf2_mgpu( ngpu, M, N, d_lA, ldda, tau, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_dgeqrf2_mgpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            printf("\"%f\",", gpu_time); // Kernel
            
            D2H_time = magma_wtime();
            magma_dgetmatrix_1D_col_bcyclic( ngpu, M, N, nb, d_lA, ldda, h_R, lda, queues );
            D2H_time = magma_wtime() - D2H_time;
            printf("\"%f\"\n", D2H_time); // D2H
            
            magma_free_cpu( tau    );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_work );
            
            magma_free_pinned( h_R    );
            
            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_free( d_lA[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_destroy( queues[dev] );
    }
 
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
