#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 64

__kernel void sum_1(__global const uint* arr,
                    __global       uint* sum,
                             const uint  n)
{
    const uint index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, arr[index]);
}

__kernel void sum_2(__global const uint* arr,
                    __global       uint* sum,
                             const uint  n)
{
    const uint index = get_global_id(0);
    uint res = 0;
    for (uint i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = index * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void sum_3(__global const uint* arr,
                    __global       uint* sum,
                             const uint  n)
{
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);
    const uint grs = get_local_size(0);

    uint res = 0;
    for (uint i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void sum_4(__global const uint* arr,
                    __global       uint* sum,
                             const uint  n)
{
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = arr[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint group_res = 0;
        for (uint i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void sum_5(__global const uint* arr,
                    __global       uint* sum,
                             const uint  n)
{
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint wid = get_group_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);


    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            uint a = buf[lid];
            uint b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}

__kernel void sum_6(__global const uint* arr,
                    __global       uint* sum,
                             const uint  n)
{
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint wid = get_group_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);


    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            uint a = buf[lid];
            uint b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (lid == 0) {
        sum[wid] = buf[0];
    }
}