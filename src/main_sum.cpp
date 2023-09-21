#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libgpu/context.h>
#include <libutils/fast_random.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    std::cout << std::endl;

    gpu::gpu_mem_32u as_gpu, sum_gpu;
    as_gpu.resizeN(n);
    sum_gpu.resizeN(1);
    
    uint zero_sum = 0;
    uint gpu_answer;

    unsigned int workGroupSize = 64;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    as_gpu.writeN(as.data(), n);
    sum_gpu.writeN(&zero_sum, 1);

    {
        timer t;
        ocl::Kernel sum_1(sum_kernel, sum_kernel_length, "sum_1");
        sum_1.compile();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_gpu.writeN(&zero_sum, 1);
            sum_1.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);

            sum_gpu.readN(&gpu_answer, 1);
            EXPECT_THE_SAME(reference_sum, gpu_answer, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_1:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_1:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        ocl::Kernel sum_2(sum_kernel, sum_kernel_length, "sum_2");
        sum_2.compile();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_gpu.writeN(&zero_sum, 1);
            sum_2.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);

            sum_gpu.readN(&gpu_answer, 1);
            EXPECT_THE_SAME(reference_sum, gpu_answer, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_2:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_2:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        ocl::Kernel sum_1(sum_kernel, sum_kernel_length, "sum_3");
        sum_1.compile();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_gpu.writeN(&zero_sum, 1);
            sum_1.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);

            sum_gpu.readN(&gpu_answer, 1);
            EXPECT_THE_SAME(reference_sum, gpu_answer, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_3:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_3:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        ocl::Kernel sum_1(sum_kernel, sum_kernel_length, "sum_4");
        sum_1.compile();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_gpu.writeN(&zero_sum, 1);
            sum_1.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);

            sum_gpu.readN(&gpu_answer, 1);
            EXPECT_THE_SAME(reference_sum, gpu_answer, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_4:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_4:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        ocl::Kernel sum_1(sum_kernel, sum_kernel_length, "sum_5");
        sum_1.compile();
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum_gpu.writeN(&zero_sum, 1);
            sum_1.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, sum_gpu, n);

            sum_gpu.readN(&gpu_answer, 1);
            EXPECT_THE_SAME(reference_sum, gpu_answer, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_5:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_5:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        gpu::gpu_mem_32u buff1, buff2;
        buff1.resizeN(n);
        buff2.resizeN(n);

        ocl::Kernel sum_1(sum_kernel, sum_kernel_length, "sum_6");
        sum_1.compile();

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            buff1.writeN(as.data(), n);
            for (uint WORK_SIZE = n; WORK_SIZE > 1; WORK_SIZE = (WORK_SIZE + workGroupSize - 1) / workGroupSize) {
                sum_1.exec(gpu::WorkSize(workGroupSize, WORK_SIZE),
                        buff1, buff2, WORK_SIZE);
                std::swap(buff1, buff2);
            }

            buff1.readN(&gpu_answer, 1);
            EXPECT_THE_SAME(reference_sum, gpu_answer, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU sum_6:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU sum_6:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

}
