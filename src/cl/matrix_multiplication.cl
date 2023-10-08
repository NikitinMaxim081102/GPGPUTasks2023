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

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE * WORK_PER_THREAD];
    float sum[WORK_PER_THREAD] = {0};


    for (uint t = 0; t < K / TILE_SIZE; t++) {
        for (int  work = 0; work < WORK_PER_THREAD; work++) {
            tileB[local_j][local_i * WORK_PER_THREAD + work] = b[(t * TILE_SIZE + local_j) * N + WORK_PER_THREAD * TILE_SIZE + i];
        }
        tileA[local_j][local_i] = a[t * TILE_SIZE + local_i + j * K];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            float tmp = tileA[local_j][k];
            for (int work = 0; work < WORK_PER_THREAD; work++) {
                sum[work] += tmp * tileB[k][local_i * WORK_PER_THREAD + work];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int work = 0; work < WORK_PER_THREAD; work++) c[j * N + WORK_PER_THREAD * TILE_SIZE + i] = sum[work];
}