#pragma OPENCL EXTENSION cl_intel_printf : enable

uint binSearch(__global float* a, const uint size, const float number, bool left) {
    uint l = -1;
    uint r = size;
    while(r - l > 1) {
        uint m = (l + r) / 2;
        if (left ? a[m] < number : a[m] <= number) l = m;
        else r = m;
    }
    return l + 1;
}

__kernel void merge(
    __global float* a, 
    __global float* b, 
    const uint blockSize,
    const uint n
) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint currentBlockBegin = i - i % blockSize;
    const int left = currentBlockBegin % (2 * blockSize) == 0 ? 0 : 1;
    uint startSearchPosition = currentBlockBegin + (1 - 2 * left) * blockSize;
    uint realPosition = binSearch(a + startSearchPosition, blockSize, a[i], (bool) left);
    // printf("%d %d %d %f %d %d %d\n", currentBlockBegin, left, startSearchPosition, a[i], i, realPosition, currentBlockBegin - left * blockSize + realPosition + i - currentBlockBegin);
    b[currentBlockBegin - left * blockSize + realPosition + i - currentBlockBegin] = a[i];
}
