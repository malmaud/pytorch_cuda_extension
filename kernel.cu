
#include "test.h"

__global__ void record_thread(float *x) {
    int idx = threadIdx.x;
    x[idx] = idx;
}

void kernel(float* data,int N) {
    record_thread<<<1, N>>>(data);
}

__global__ void jsum(float *a, float *b, float *c, float *d, int N) {
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    while(idx<N) {
        d[idx] = a[idx]*b[idx]+c[idx];
        idx += num_threads;
    }
}

void jsum_host(float *a, float* b,float *c,float*d, int N) {
    jsum<<<(N/256)+1, 256>>>(a,b,c,d,N);
}
