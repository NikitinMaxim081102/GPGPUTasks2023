#define TILE_SIZE 16
#define COUNT 4
#pragma OPENCL EXTENSION cl_intel_printf : enable
#line 5

__kernel void reduce(__global uint *as, const uint shift, const uint size) {
    const uint i = get_global_id(0);
    const uint begin = (i + 1) * 2 * shift - 1;
    if (begin >= size || begin - shift >= size) return;
    as[begin] += as[begin - shift];
}

__kernel void scan(__global uint *as, const uint shift, const int deleteLast, const uint size) {
    const int i = get_global_id(0);
    if (i == 0 && deleteLast) as[size - 1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint begin = (i + 1) * 2 * shift - 1;
    if (begin >= size || begin - shift >= size) return;
    uint tmp = as[begin];
    as[begin] += as[begin - shift];
    as[begin - shift] = tmp;
}

uint get_number(uint big_number, uint shift) {
    return (big_number >> (shift * 2)) & (COUNT - 1);
}

uint binSearch(__global uint* a, const uint size, const uint number, bool left, const uint shift) {
    uint l = -1;
    uint r = size;
    while(r - l > 1) {
        uint m = (l + r) / 2;
        if (left ? get_number(a[m], shift) <= number : get_number(a[m], shift) < number) l = m;
        else r = m;
    }
    return l + 1;
}

__kernel void merge(
    __global const uint* a, 
    __global       uint* b, 
    const uint blockSize,
    const uint n,
    const uint shift
) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint currentBlockBegin = i - i % blockSize;
    const int left = currentBlockBegin % (2 * blockSize) == 0 ? 0 : 1;
    uint startSearchPosition = currentBlockBegin + (1 - 2 * left) * blockSize;
    uint realPosition = binSearch(a + startSearchPosition, blockSize, get_number(a[i], shift), (bool) left, shift);
    // printf("%d %d %d %d %d %d %d\n", currentBlockBegin, left, startSearchPosition, get_number(a[i], shift), i, realPosition, currentBlockBegin - left * blockSize + realPosition + i - currentBlockBegin);
    b[currentBlockBegin - left * blockSize + realPosition + i - currentBlockBegin] = a[i];
}

__kernel void matrix_transpose(
    __global const uint* a,
    __global       uint* at,
    const uint size
    // const uint n
) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    for (int i = 0; i < COUNT; i++) {
        at[i * size + gid] = a[gid * COUNT + i];
    }
}

__kernel void get_counts_table(
    __global const uint *as,
    __global       uint* cs,
    uint shift,
    const uint n
) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint wid = get_group_id(0);
    if (gid >= n) return;
    __local uint local_count[COUNT];
    if (lid < COUNT) {
        local_count[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&local_count[get_number(as[gid], shift)], 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < COUNT) {
        cs[wid * COUNT + lid] = local_count[lid];
    }
}

__kernel void move(
    __global const uint *as,
    __global       uint* cs,
    uint size,
    uint last,
    const uint n
) {
    uint gid = get_global_id(0);
    if (gid >= n) return;
    uint value = as[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid > 0)
        cs[gid - 1] = value;
    if (gid == size - 1)
        cs[gid] = last;
}

__kernel void radix(
    __global const uint *as,
    __global       uint *bs,
    __global const uint *prefSumT,
    __global const uint *prefSum,
    uint number_of_groups,
    uint shift,
    const uint n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int grid = get_group_id(0);
    unsigned int number = get_number(as[gid], shift);
    int realPosition = get_local_id(0) - prefSum[grid * COUNT + number];
    if (grid || number) {
        realPosition += prefSumT[number * number_of_groups + grid - 1];
    }
    bs[realPosition] = as[gid];
}

__kernel void prefix_sum(
    __global const uint *as,
    __global       uint *bs,
    const uint n
) {
    const uint gid = get_global_id(0);
    if (gid >= n) return;
    uint sum = 0;
    bs[gid * COUNT] = 0;
    for(int i = 1; i < COUNT; i++) {
        sum += as[gid * COUNT + i - 1];
        bs[gid * COUNT + i] = sum;
    }
}
