#line 2

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