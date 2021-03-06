//ly2352, Lu Yang, Adaboost, Host version
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
int nums = 2000,cols = 256;
float **usps;
float *w;
int *y;

struct pars{
    int return_j;
    float theta;
    int return_m;
};

void train(struct pars* pars_p){
  int cur_j = 0,cur_theta = 0,cur_m = 0;
  float cur_min = 100000.0;
  for (int j = 0;j<cols;j++){
    float *vec = usps[j];   //vec = as.vector(X[j])
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
    struct pars* allPars = malloc(sizeof(struct pars)*B);
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
    float *res = malloc(sizeof(float)*nums);
    for (int z = 0; z < nums; ++z)
        res[z] = 0.0;
    int *c_hat = malloc(sizeof(int)*nums);
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
    clock_t begin, end;
    double time_spent;
    begin = clock();
    struct pars* ap;
    float *alpha = malloc(sizeof(float)*5);;
    int *c_hat;
    ap = AdaBoost(5,alpha);
    c_hat = agg_class(alpha,ap,5);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    for (int i = 0; i < 5; ++i)
    {
        printf("%d,%f,%d,%f\n",ap[i].return_j,ap[i].theta,ap[i].return_m,alpha[i]);    
    }
    printf("time is %f\n",time_spent);
    for(int j=0;j<cols;j++){
        free(usps[j]);
    }
    free(usps);
    free(w);
    free(y);
    free(alpha);
    free(ap);
    free(c_hat);
    return 0;
}
