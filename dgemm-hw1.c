/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
const char* dgemm_desc = "Custom dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

#define CACHELINESIZE 64
#define PAGESIZE 4096


double* align_B(int lda, double* B, int allocate_dim)
{
	//int allocate_dim = (int) ceil(lda * sizeof(double) / (double) CACHELINESIZE) * CACHELINESIZE;
	//int allocate_size = (int) ceil(allocate_dim * (double)lda /PAGESIZE)*PAGESIZE;
	
	int allocate_size = allocate_dim*lda*sizeof(double);
	//printf("allocate size: %d\n", allocate_size);
        
	int npages = (int) ceil(allocate_size/(double)PAGESIZE);
	//printf("npages: %d\n", npages);

	
	//double* B_prime = (double *) calloc(allocate_size/sizeof(double), sizeof(double));

	double* B_prime = (double *) mmap(0, npages*PAGESIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	//printf("B_prime: %lx\n", (long) B_prime);
	//printf("errno %d\n", errno);
	
	for (int i = 0; i < lda; i++) {
		memcpy( B_prime + i*allocate_dim, B + i*lda, lda * sizeof(double));
	}
	
	return B_prime;
}

double* align_A(int lda, double* A, int allocate_dim)
{
	int allocate_size = allocate_dim*lda*sizeof(double);
	int npages = (int) ceil(allocate_size/(double)PAGESIZE);
	double* A_prime = (double *) mmap(0, npages*PAGESIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	
	for(int i = 0; i < lda; i++)
	{
		for (int j = 0; j < lda; j++)
		{
			A_prime[i+j*allocate_dim] = A[i*lda+j];
		}
	}
	return A_prime; 
}

void square_dgemm(int lda, double* A, double* B, double* C)
{
	int allocate_dim = (int) (ceil(lda * sizeof(double) / (double) CACHELINESIZE) * CACHELINESIZE)/sizeof(double);
	double* A_prime = align_A(lda, A, allocate_dim);
	double* B_prime = align_B(lda, B, allocate_dim);
	for (int j = 0; j < lda; j++)
	{
		for (int i = 0; i<lda; i++)
		{
			double cij = C[i+j*lda];
			for(int k = 0; k<lda; k++)
			{
				cij += A_prime[i*allocate_dim+k]*B_prime[j*allocate_dim+k];
			}
			C[i+j*lda] = cij;
		}
	}	
}



/*
int main()
{
	int lda = 3;
	double test_matrix[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
	int allocate_dim = (int) (ceil(lda * sizeof(double) / (double) CACHELINESIZE) * CACHELINESIZE)/sizeof(double);
	printf("allocate_dim: %d\n", allocate_dim);
	double* B_prime = align_B(3, test_matrix, allocate_dim);
	double* A_prime = align_A(3, test_matrix, allocate_dim);	

	printf("Printing Bprime\n");
	for(int j = 0; j < lda; j++)
	{
		for(int i = 0; i< allocate_dim; i++)
		{
			printf("%d %d %f \n", j, i, B_prime[i + j*allocate_dim]);
		}
	}

	printf("Printing A prime\n");
        for(int j = 0; j < lda; j++)
        {
                for(int i = 0; i< allocate_dim; i++)
                {
                        printf("%d %d %f \n", j, i, A_prime[i + j*allocate_dim]);
                }
        }

	printf("Printing Test matrix\n");

	for(int j = 0; j < lda; j++)
        {
                for(int i = 0; i< lda; i++)
                {
                        printf("%d %d %f \n", j, i, test_matrix[i + j*lda]);
                }
        }

	return 0;
}*/
