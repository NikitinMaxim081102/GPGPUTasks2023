#define TILE_SIZE 16

__kernel void matrix_transpose(
    __global const float* a,
    __global float* at, 
    const uint m, 
    const uint k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (!(i < k && j < m)) return;
    tile[local_i][local_j] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    float tmp = tile[local_i][local_j];

    barrier(CLK_LOCAL_MEM_FENCE);

    tile[local_i][local_j] = tile[local_j][local_i];
    tile[local_j][local_i] = tmp;

    at[i * m + j] = tile[local_j][local_i];

}