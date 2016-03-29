//ly2352, Lu Yang, Adaboost, Host version
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#define nums 200
#define cols 256
//int nums = 200,cols = 256;
float **usps;
float *w;
float *d_w;
int *y;
int *d_y;
float *d_vec, *d_err_diff;

struct pars{
    int return_j;
    float theta;
    int return_m;
};

void cuda_checker(cudaError_t err){
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    __shared__ float sum[nums];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        sum[i] = A[i] + B[i];
    }
    __syncthreads();
    float tmp = 0.0;
    for (int i = 0; i < nums; ++i){
        tmp+=sum[i];
    }
    *C = tmp;
}
/*
__global__ void
vectorAdd_train(const float *vec, const float *w, const int *y,
    float *err_diff, int numElements, float boundary){

    int z = blockDim.x * blockIdx.x + threadIdx.x;
    if (z < numElements)
    {
        err_diff[z] = w[z] * ( ((vec[z]<=boundary) ^ (y[z]==-1)) -  ((vec[z]<=boundary) ^ (y[z]==1)));
    }
}

void cuda_train(struct pars* pars_p){
    size_t size = nums * sizeof(float);
    cuda_checker(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    float *err_diff = (float*)malloc(size);
    int cur_j = 0,cur_theta = 0,cur_m = 0;
    float cur_min = 100000.0;
    for (int j = 0;j<cols;j++){
        float *vec = usps[j];
        cuda_checker(cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice));
        float minimal = 100000.0;
        int cur_i = 0,sel_m= 0;
        for(int i=0;i<nums;i++){
            float boundary = vec[i];
            int m = 0;
            //float err = 0.0,err1 = 0.0,err2 = 0.0,sum_w = 0;

            int threadsPerBlock = 256;
            int blocksPerGrid =(nums + threadsPerBlock - 1) / threadsPerBlock;
            vectorAdd_train<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_w, d_y,
                d_err_diff, numElements,boundary);
            cuda_checker(cudaMemcpy(err_diff, d_err_diff, size, cudaMemcpyDeviceToHost));
            float err_sum = 0.0,w_sum = 0.0;
            for (int i = 0; i < nums; ++i)
            {
                err_sum += err_diff;
                w_sum += w[i];
            }
            err = err_sum>0?
            if(err1<err2){
                err = err1/sum_w;
                m = 1;
            }else{
                err = err2/sum_w;
                m = -1;
            }
            if(err<minimal){
                minimal = err;
                cur_i = i;
                sel_m = m;
            }
        }
    if(minimal<cur_min){
        cur_min = minimal;
        cur_j = j;
        cur_theta = cur_i;
        cur_m = sel_m;
    }
  }
  pars_p->return_j = cur_j;
  pars_p->theta = usps[cur_j][cur_theta];
  pars_p->return_m = cur_m;
  free(err_diff);
  return;
}
*/
void train(struct pars* pars_p){
  int cur_j = 0,cur_theta = 0,cur_m = 0;
  float cur_min = 100000.0;
  for (int j = 0;j<cols;j++){
    float *vec = usps[j];
    float minimal = 100000.0;
    int cur_i = 0,sel_m= 0;
    for(int i=0;i<nums;i++){
      float boundary = vec[i];
      float err = 0.0,err1 = 0.0,err2 = 0.0,sum_w = 0;
      int m = 0;
      for(int z=0;z<nums;z++){
        err1 += w[z] * ((vec[z]<=boundary) ^ (y[z]==-1));
        err2 += w[z] * ((vec[z]<=boundary) ^ (y[z]==1));
        sum_w += w[z];
      }
      
      if(err1<err2){
        err = err1/sum_w;
        m = 1;
      }else{
        err = err2/sum_w;
        m = -1;
      }
      if(err<minimal){
        minimal = err;
        cur_i = i;
        sel_m = m;
      }
    }
    if(minimal<cur_min){
      cur_min = minimal;
      cur_j = j;
      cur_theta = cur_i;
      cur_m = sel_m;
    }
  }
  pars_p->return_j = cur_j;
  pars_p->theta = usps[cur_j][cur_theta];
  pars_p->return_m = cur_m;
  return;
}

struct pars* AdaBoost(int B,float *alpha){
    struct pars* allPars = (struct pars*)malloc(sizeof(struct pars)*B);
    for (int b=0;b<B;b++){
        struct pars pars;
        train(&pars);
        // label = classify(X,pars)
        float *vec = usps[pars.return_j];
        float err = 0.0,w_sum = 0.0;
        for(int z =0;z<nums;z++){
            err += w[z] * ((vec[z]<=pars.theta) ^ (-pars.return_m == y[z]) );
            w_sum += w[z];
        }
        err = err/w_sum;
        alpha[b] = logf((1-err)/err);
        for(int z =0;z<nums;z++){
            w[z] = ((vec[z]<=pars.theta) ^ (-pars.return_m == y[z]))?(w[z] * (1-err) / err):w[z];
        }
        allPars[b].return_j = pars.return_j;
        allPars[b].return_m = pars.return_m;
        allPars[b].theta = pars.theta;
    }
    return allPars;
}

int * agg_class(float *alpha,struct pars* allPars,int B){
    float *res = (float *)malloc(sizeof(float)*nums);
    for (int z = 0; z < nums; ++z)
        res[z] = 0.0;
    int *c_hat = (int *)malloc(sizeof(int)*nums);
    for (int b = 0; b < B; ++b)
    {
        struct pars pars = allPars[b];
        float *vec = usps[pars.return_j];
        for(int z=0;z<nums;z++){
            res[z] += alpha[b]* ((vec[z]<=pars.theta)?(-pars.return_m):pars.return_m);
        }
    }
    for (int z = 0; z < nums; ++z)
        c_hat[z] = res[z]>= 0 ? 1:-1;
    free(res);
    return c_hat;
}

int main(){
    usps = (float **)malloc(sizeof(float *)*cols);
    w = (float*)malloc(sizeof(float)*nums);
    y = (int*)malloc(sizeof(int)*nums);;
    for (int i = 0; i < nums; ++i){
        w[i] = 1.0/nums;    
    }
    for(int j=0;j<cols;j++){
        usps[j] = (float *)malloc(sizeof(float)*nums);
    } 
    FILE* fp = fopen("uspsdata/uspsdata_ext.txt","r");
    FILE* fpcl = fopen("uspsdata/uspscl_ext.txt","r");
    for(int i=0;i<nums;i++){
        fscanf(fpcl,"%d",y+i);
        for(int j=0;j<cols;j++){
            fscanf(fp,"%f",*(usps+j)+i);
        }
    }
    fclose(fp);fclose(fpcl);

    /***********cuda here********/
    cudaError_t err = cudaSuccess;
    int numElements = nums;
    size_t size = numElements * sizeof(float);

    float *h_A = usps[0];
    float *h_B = usps[1];
    float *h_C = (float *)malloc(size);
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    cuda_checker(cudaMalloc((void **)&d_w, size));
    cuda_checker(cudaMalloc((void **)&d_err_diff, size));
    cuda_checker(cudaMalloc((void **)&d_vec, size));
    cuda_checker(cudaMalloc((void **)&d_y, nums*sizeof(int)));

    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    cuda_checker(err);

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    cuda_checker(err);

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    cuda_checker(err);

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_y, y, sizeof(int)*nums, cudaMemcpyHostToDevice);
    cuda_checker(err);
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cuda_checker(err);

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cuda_checker(err);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    err = cudaGetLastError();
    cuda_checker(err);

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cuda_checker(err);

    for (int i = 0; i < nums; ++i)
    {
        printf("float is %f\n", h_C[i]);
    }
    
    cuda_checker( cudaFree(d_A));
    cuda_checker( cudaFree(d_B));
    cuda_checker( cudaFree(d_C));
    cuda_checker( cudaFree(d_w));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(h_C);
    cuda_checker(cudaDeviceReset());
    /*****************************/


    // clock_t begin, end;
    // double time_spent;
    // begin = clock();
    // struct pars* ap;
    // float *alpha = (float *)malloc(sizeof(float)*5);;
    // int *c_hat;
    // ap = AdaBoost(5,alpha);
    // c_hat = agg_class(alpha,ap,5);
    // end = clock();
    // time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    // for (int i = 0; i < 5; ++i)
    // {
    //     printf("%d,%f,%d,%f\n",ap[i].return_j,ap[i].theta,ap[i].return_m,alpha[i]);    
    // }
    // printf("time is %f\n",time_spent);
    // for(int j=0;j<cols;j++){
    //     free(usps[j]);
    // }
    // free(usps);
    // free(w);
    // free(y);
    // free(alpha);
    // free(ap);
    // free(c_hat);
    return 0;
}
