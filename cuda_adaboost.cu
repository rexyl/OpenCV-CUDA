//ly2352, Lu Yang, Adaboost, Host version
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#define nums 20
#define cols 256
//int nums = 200,cols = 256;
float **usps;
float *w;
float *d_w;
float *d_sum_w;
int *y;
int *d_y;
float *d_vec, *d_err1, *d_err2;

struct pars{
    int return_j;
    float theta;
    int return_m;
};

void cuda_checker(cudaError_t err){
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements){
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

__global__ void
vectorAdd_train(const float *vec, const float *w, const int *y,
    float *err1, float *err2, float *sum_w,int numElements, float boundary){
    __shared__ float sum1[nums];
    __shared__ float sum2[nums];

    int z = blockDim.x * blockIdx.x + threadIdx.x;
    if (z < numElements)
    {
        sum1[z] = w[z] * ((vec[z]<=boundary) ^ (y[z]==-1));
        sum2[z] = w[z] * ((vec[z]<=boundary) ^ (y[z]==1));
    }
    __syncthreads();
    float tmp1 = 0.0 , tmp2 = 0.0 , tmp3 = 0.0;
    for (int i = 0; i < nums; ++i){
        tmp1+=sum1[i];
        tmp2+=sum2[i];
        tmp3+=w[i];
    }
    *err1 = tmp1;
    *err2 = tmp2;
    *sum_w = tmp3;
}

__global__ void
vectorAdd_train2d(const float *vec, const float *w, const int *y,
    float *min_out,int * cur_i_out,int * sel_m_out,int numElements){
    float sum1[nums];    //err1[]
    float sum2[nums];    //err2[]
    __shared__ float minimal[nums];
    __shared__ int m[nums];
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.x * blockIdx.x + threadIdx.x;
    printf("[%d,%d]\n",i,z );
    float boundary = vec[i];
    if (z < numElements && i < numElements)
    {
        sum1[z] = w[z] * ((vec[z]<=boundary) ^ (y[z]==-1));
        sum2[z] = w[z] * ((vec[z]<=boundary) ^ (y[z]==1));
    }
    else{
        return;
    }
    __syncthreads();
    if (z < numElements && i < numElements){
        float tmp1 = 0.0 , tmp2 = 0.0 , tmp3 = 0.0;
        float err1,err2;
        for (int t = 0; t < nums; ++t){
            tmp1+=sum1[t];
            tmp2+=sum2[t];
            tmp3+=w[t];
        }
        err1 = tmp1/tmp3;
        err2 = tmp2/tmp3;
        //printf("err1 is %f,err2 is %f\n",err1,err2 );
        minimal[i] = err1<err2?err1:err2;
        m[i] = err1<err2?1:-1;
    }
    __syncthreads();
    if (z == 0 && i == 0)
    {
        float min_tmp = 100000.0;
        int cur_i = -1, sel_m = 0;
        for (int t = 0; t < nums; ++t)
        {
            //printf("min_tmp:%f,minimal[t]:%f\n",min_tmp,minimal[t] );
            cur_i = min_tmp<minimal[t]?cur_i:t;
            sel_m = min_tmp<minimal[t]?sel_m:m[t];
            min_tmp = min_tmp<minimal[t]?min_tmp:minimal[t];
        }
        *min_out = min_tmp;
        *sel_m_out = sel_m;
        *cur_i_out = cur_i;
        printf("min_out is %f,sel_m_out %d,cur_i_out %d\n",min_tmp,sel_m,cur_i);
    }
    
    
}

void cuda_train1(struct pars* pars_p){
    size_t size = nums * sizeof(float);
    cuda_checker(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    int cur_j = 0,cur_theta = 0,cur_m = 0;
    float cur_min = 100000.0;
    for (int j = 0;j<cols;j++){
        float *vec = usps[j];
        cuda_checker(cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice));
        float minimal = 100000.0;
        int cur_i = 0,sel_m= 0;
        dim3 block(16,16);
        dim3 grid ((nums+15)/16,(nums+15)/16);

        float *min_out;
        cuda_checker(cudaMalloc((void **)&min_out,sizeof(float)));
        int *cur_i_out,*sel_m_out;
        cuda_checker(cudaMalloc((void **)&cur_i_out,sizeof(int)));
        cuda_checker(cudaMalloc((void **)&sel_m_out,sizeof(int)));
        vectorAdd_train2d<<<grid,block>>>(d_vec,d_w,d_y,min_out,cur_i_out,sel_m_out,nums);
        
        cuda_checker(cudaMemcpy(&minimal, min_out, sizeof(float), cudaMemcpyDeviceToHost));
        cuda_checker(cudaMemcpy(&cur_i, cur_i_out, sizeof(int), cudaMemcpyDeviceToHost));
        cuda_checker(cudaMemcpy(&sel_m, sel_m_out, sizeof(int), cudaMemcpyDeviceToHost));
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
  //printf("%d,%f,%d",cur_j,pars_p->theta, cur_m);
  return;
}
/*
void cuda_train(struct pars* pars_p){
    size_t size = nums * sizeof(float);
    cuda_checker(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    float err1,err2,sum_w,err;
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
            int threadsPerBlock = 256;
            int blocksPerGrid =(nums + threadsPerBlock - 1) / threadsPerBlock;
            vectorAdd_train<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_w, d_y,
                d_err1, d_err2, d_sum_w,nums,boundary);
            cuda_checker(cudaMemcpy(&err1, d_err1, sizeof(float), cudaMemcpyDeviceToHost));
            cuda_checker(cudaMemcpy(&err2, d_err2, sizeof(float), cudaMemcpyDeviceToHost));
            cuda_checker(cudaMemcpy(&sum_w, d_sum_w, sizeof(float), cudaMemcpyDeviceToHost));
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
        cuda_train1(&pars);
        //train(&pars);
        //cuda_train(&pars);
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
    int numElements = nums;
    size_t size = numElements * sizeof(float);
    cuda_checker(cudaMalloc((void **)&d_w, size));
    //cuda_checker(cudaMalloc((void **)&d_sum_w, sizeof(float)));
    //cuda_checker(cudaMalloc((void **)&d_err1, sizeof(float)));
    //cuda_checker(cudaMalloc((void **)&d_err2, sizeof(float)));
    cuda_checker(cudaMalloc((void **)&d_vec, size));
    cuda_checker(cudaMalloc((void **)&d_y, nums*sizeof(int)));

    cuda_checker(cudaMemcpy(d_y, y, sizeof(int)*nums, cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    // vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // err = cudaGetLastError();
    // cuda_checker(err);

    // for (int i = 0; i < nums; ++i)
    // {
    //     printf("float is %f\n", h_C[i]);
    // }
    /*****************************/
    clock_t begin, end;
    double time_spent;
    begin = clock();
    struct pars* ap;
    float *alpha = (float *)malloc(sizeof(float)*5);;
    int *c_hat;
    //ap = AdaBoost(5,alpha);
    //c_hat = agg_class(alpha,ap,5);
    struct pars pars;
    cuda_train1(&pars);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    // for (int i = 0; i < 5; ++i)
    // {
    //     printf("%d,%f,%d,%f\n",ap[i].return_j,ap[i].theta,ap[i].return_m,alpha[i]);    
    // }
    //printf("time is %f\n",time_spent);
    for(int j=0;j<cols;j++){
        free(usps[j]);
    }
    free(usps);
    free(w);
    free(y);
    free(alpha);
    //free(ap);
    //free(c_hat);

    cuda_checker(cudaFree(d_w));
    //cuda_checker(cudaFree(d_sum_w));
    //cuda_checker(cudaFree(d_err1));
    //cuda_checker(cudaFree(d_err2));
    cuda_checker(cudaFree(d_vec));
    cuda_checker(cudaFree(d_y));
    cuda_checker(cudaDeviceReset());
    return 0;
}