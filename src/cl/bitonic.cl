#pragma OPENCL EXTENSION cl_intel_printf : enable

__kernel void bitonic(__global float *as, const uint count, const uint shift) {
    const int i = get_global_id(0);
    bool reverse = (i / (count / 2)) % 2 == 1;
    uint begin = (i / count) * count * 2;
    uint id = i % count;
    begin += (id / shift) * shift * 2 + id % shift;

    // printf("%d %d %d %f %f\n", i, reverse, begin, as[begin], as[begin + shift]);
    if (reverse) {
        if (as[begin] < as[begin + shift]) {
            float tmp = as[begin];
            as[begin] = as[begin + shift];
            as[begin + shift] = tmp;
        }
    } else {
        if (as[begin] > as[begin + shift]) {
            float tmp = as[begin];
            as[begin] = as[begin + shift];
            as[begin + shift] = tmp;
        }
    }

}
