#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <limits.h>

#define nums 200000

int cmpfunc (const void * a, const void * b){
   return ( *(int*)a - *(int*)b );
}
int kthSmallest(int arr[], int l, int r, int k){
    return qsort(arr, nums, sizeof(int), cmpfunc);
}

void cuda_checker(cudaError_t err,int i){
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device(error code %s and %d)!\n", cudaGetErrorString(err),i);
        exit(EXIT_FAILURE);
    }
}
__global__ void
cuda_kthsmall(const int *x,const int k){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nums)
    {	
    	int upper_sum = 0,down_sum = 0,pivot = x[idx];
	    for (int i = 0; i < nums; ++i)
	    {
	    	upper_sum += (pivot>x[i]);
	    	down_sum += (pivot>=x[i]);
	    }
	    if (k<=down_sum && k>upper_sum)
	    {
	    	printf("Found, %d\n",pivot);
	    }
    }
}


int main(){
    int *x = (int*)malloc(sizeof(int)*nums);
    time_t t;
    clock_t begin,end;
    double time_spend;
    int err_num = 0;
    srand((unsigned) time(&t));
    
    for (int i = 0; i < nums; ++i)
    {
        x[i] = rand() % 1000;
        //printf("%d ",x[i]);
    }
    printf("\n");
  
    begin = clock();
    printf("%d\n", kthSmallest(x,0,nums-1,3));
    end = clock();
    time_spend = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU cost %f\n", time_spend);

    int *d_x = NULL;
    begin = clock();
    cuda_checker(cudaMalloc((void **)&d_x,sizeof(int)*nums),err_num++);
    cuda_checker(cudaMemcpy(d_x, x, sizeof(int)*nums, cudaMemcpyHostToDevice),err_num++);
    cuda_kthsmall<<<(nums + 256 - 1) / 256, 256>>>(d_x,3);
    cuda_checker(cudaFree(d_x),err_num++);
    end = clock();
    time_spend = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU cost %f\n", time_spend);

    free(x);
    cuda_checker(cudaDeviceReset(),err_num++);
    return 0;
}