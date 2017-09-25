#include "mat_mult.h"
extern "C"
{

#define MW_TITLE_SIZE 96
#define MW_TRUNK_SIZE 16 

/********************************************************************
* compute matrix BATCH_IN * WEIGHT_IN
* Args :
*   V_DIM : int vector_dim
*   BATCH_SIZE : pad_query batch size
*   OUT_DIM : output dim
*   BATCH_IN : a  V_DIM x BATCH_SIZE dim matrix for batch query
*   WEIGHT_IN : a V_DIM x ouput_num dim matrix for big wight
*   OUT : out matrix  is BATCH_SIZE x OUT_DIM dim 
*/
__global__ void kComputeMatMult(const int V_DIM, const int BATCH_SIZE, const int OUT_DIM,
                                  const float* BATCH_IN, const float* WEIGHT_IN,
                                  float* OUT) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = ty * 16 + tx;

  //printf("GPU from thread with tid %d\n" , tid);
  int tx2 = tid % 32;
  int ty2 = tid / 32;

  volatile __shared__ float as[16][96]; // 16 block size , 96 ,query_batch size
  volatile __shared__ float bs[16][96];

  float cr[6][6];
  float ar;
  float br[6];

  float asr[2][3];
  float bsr[2][3];

  //for (int i=0;i<10;++i)
  //{
  //  printf("Batch[%d] = %f\n" , i, BATCH_IN[i]);
  //  printf("Weight[%d] = %f\n" , i, WEIGHT_IN[i]);
  //}

  BATCH_IN += ty2 * BATCH_SIZE + (by * 96 + tx2);
  //printf("Batch address In %d and batch value %f\n" , BATCH_IN,*BATCH_IN);

  WEIGHT_IN += ty2 * OUT_DIM + (bx * 96 + tx2);
  //printf("Weight address In %d weight value %f\n" , WEIGHT_IN,WEIGHT_IN[0]);
  OUT += (by * 96 + ty) * OUT_DIM + (bx * 96 + tx);

  // Zero OUT reg
  #pragma unroll
  for (int i = 0; i < 6; ++i)
    #pragma unroll
    for (int j = 0; j < 6; ++j) cr[i][j] = 0.0f;

  // Load BATCH_IN gmem->smem
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 3; ++j) as[i * 8 + ty2][j * 32 + tx2] = BATCH_IN[j * 32];
    BATCH_IN += BATCH_SIZE * 8;
  }

  // Load WEIGHT_IN gmem->smem
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 3; ++j) bs[i * 8 + ty2][j * 32 + tx2] = WEIGHT_IN[j * 32];
    WEIGHT_IN += OUT_DIM * 8;
  }

  __syncthreads();

  for (int kk = 0; kk < V_DIM - 16; kk += 16) {
    // Load BATCH_IN gmen->reg
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) asr[i][j] = BATCH_IN[j * 32];
      BATCH_IN += BATCH_SIZE * 8;
    }

    // Load WEIGHT_IN gmem->reg
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) bsr[i][j] = WEIGHT_IN[j * 32];
      WEIGHT_IN += OUT_DIM * 8;
    }

    // Compute
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      // Load WEIGHT_IN smen->reg
      #pragma unroll
      for (int j = 0; j < 6; ++j) br[j] = bs[k][j * 16 + tx];

      #pragma unroll
      for (int i = 0; i < 6; ++i) {
        ar = as[k][i * 16 + ty];
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
          float d = ar * br[j]; //ar - br[j];
          //printf("tid %d : ar %f * br[%d] %f is d %f\n",tid, ar,j,br[j],d);
          cr[i][j] += d; //d * d;
        }
      }
    }

    __syncthreads();

    // Load BATCH_IN reg->smem
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 3; ++j) as[i * 8 + ty2][j * 32 + tx2] = asr[i][j];

    // Load WEIGHT_IN reg->smem
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 3; ++j) bs[i * 8 + ty2][j * 32 + tx2] = bsr[i][j];

    __syncthreads();
  }

  // Compute last 16 dimensions
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
    // Load WEIGHT_IN smen->reg //share memory -> register
    #pragma unroll
    for (int j = 0; j < 6; ++j) br[j] = bs[k][j * 16 + tx];

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
      ar = as[k][i * 16 + ty];
      #pragma unroll
      for (int j = 0; j < 6; ++j) {
        float d = ar * br[j]; //ar - br[j];
        cr[i][j] += d; //d * d;
      }
    }
  }

  // Store OUT reg->gmem // register -> global memory
  #pragma unroll
  for (int i = 0; i < 6; ++i) {
    #pragma unroll
    for (int j = 0; j < 6; ++j) {
      //long long c = (long long)__float_as_int(cr[i][j]);
      //c = (c<< 32) | (bx * 96 + j * 16 + tx);
      //printf("tid %d : cr[%d][%d] = %lld\n" ,tid,i,j,cr[i][j]);
      OUT[j * 16] = cr[i][j];
    }
    OUT += OUT_DIM * 16;
  }
}

/**
int main(){
    printf("call from mat_mult in c\n");
    int K = 128;
    int M = 96;
    int N = 96;
    float A[128*96];
    for(int i=0;i<K*M;i++) A[i] = i;
    float B[128*96];
    for(int i=0;i<K*N;i++) B[i] = i;

    float C[96*96] = {0} ;
    int size = K * M * sizeof(float); 
    float * d_A;
    cudaMalloc(&d_A, size); 
    float* d_B;
    cudaMalloc(&d_B, size); 
    long long* d_C;
    cudaMalloc(&d_C, M*N*sizeof(long long)); 

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); 

    dim3 grid(1,1);
    dim3 block(16,16);

    kComputeDistances<<<grid,block>>>(K, M, N, d_A, d_B, d_C );

    cudaMemcpy(C,d_C,M*N*sizeof(long long), cudaMemcpyDeviceToHost);
    for(int i=0;i<96;i++)
        printf("C[%d] = %f\n", i, C[i]);

    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 
    return 0;
}
*/
}
