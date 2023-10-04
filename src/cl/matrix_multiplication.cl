#define TILE_SIZE 16
#define WORK_PER_THREAD 4

__kernel void matrix_multiplication_1(
    __global const float* a, 
    __global const float* b, 
    __global float* c, 
    unsigned int M, 
    unsigned int K, 
    unsigned int N
) {
    int i = get_global_id(0) / N;
    int j = get_global_id(0) % N;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_2(
    __global const float* a, 
    __global const float* b, 
    __global float* c, 
    unsigned int M, 
    unsigned int K, 
    unsigned int N
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; TILE_SIZE * tileK < K; tileK++) {
        tileA[local_j][local_i] = a[j * K + tileK * TILE_SIZE + local_i];
        tileB[local_j][local_i] = b[(local_j + TILE_SIZE * tileK) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_3(
    __global const float* a, 
    __global const float* b, 
    __global float* c, 
    unsigned int M, 
    unsigned int K, 
    unsigned int N
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE * WORK_PER_THREAD][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    float sum[WORK_PER_THREAD] = {0};

    for (int tileK = 0; TILE_SIZE * tileK < K; tileK++) {
        for (int work = 0; work < WORK_PER_THREAD; work++) {
            tileA[local_j * WORK_PER_THREAD + work][local_i] = a[(j * WORK_PER_THREAD + work) * K + tileK * TILE_SIZE + local_i];
        }
        tileB[local_j][local_i] = b[(local_j + TILE_SIZE * tileK) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            float tmp = tileB[k][local_i];
            for (int work = 0; work < WORK_PER_THREAD; work++) {
                sum[work] += tileA[local_j * WORK_PER_THREAD + work][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int work = 0; work < WORK_PER_THREAD; work++) c[(j * WORK_PER_THREAD + work) * N + i] = sum[work];
}