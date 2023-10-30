#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_map>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    // int benchmarkingIters = 1;
    // unsigned int n = 32 * 1024 * 1024;
    unsigned int n = 1024 * 1024;
    // unsigned int n = 64;
    std::unordered_map<int, int> counts;
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> bs(n, 0);

    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        // as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
        as[i] = (unsigned int) r.next(0, 8);
        counts[as[i]]++;
        bs[i] = as[i];
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    int workGroupSize = 256;
    // int workGroupSize = 32;
    uint groupsCount = n / workGroupSize;
    
    // std::unordered_map<int, int> counts;
    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u tmp;
    gpu::gpu_mem_32u counts_gpu;
    gpu::gpu_mem_32u counts_T_gpu;
    gpu::gpu_mem_32u pref_gpu;
    gpu::gpu_mem_32u counts_pref_sum_gpu;
    std::vector<unsigned int> test(groupsCount * 4, 0);
    std::vector<unsigned int> test2(n, 0);
    uint res;
    as_gpu.resizeN(n);
    tmp.resizeN(n);
    counts_gpu.resizeN(n * 4);
    counts_T_gpu.resizeN(n * 4);
    pref_gpu.resizeN(n * 4);
    counts_pref_sum_gpu.resizeN(n * 4);


    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
        reduce.compile();

        ocl::Kernel scan(radix_kernel, radix_kernel_length, "scan");
        scan.compile();

        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge");
        merge.compile();

        ocl::Kernel get_counts_table(radix_kernel, radix_kernel_length, "get_counts_table");
        get_counts_table.compile();

        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel move(radix_kernel, radix_kernel_length, "move");
        move.compile();

        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        prefix_sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            // for (auto v: bs) {
            //         std::cout << v << " ";
            // }
            // std::cout << std::endl;
            // for (uint shift = 0; shift < 32 / 2; shift++) {
            for (uint shift = 0; shift < 4; shift++) {
                // std::unordered_map<int, int> testCounts;
                // if (shift > 0) return 0;
                for (uint blockSize = 1; blockSize <= workGroupSize / 2; blockSize *= 2) {
                    // std::unordered_map<int, int> testCounts2;
                    merge.exec(gpu::WorkSize(workGroupSize, n), as_gpu, tmp, blockSize, n, shift);
                    std::swap(as_gpu, tmp);
                    // std::cout << "merge blockSize = " << blockSize << std::endl;
                    // as_gpu.readN(test2.data(), n);
                    // for (auto v: test2) {
                    //     testCounts2[v]++;
                    //     std::cout << v << " ";
                    // }
                    // std::cout << "counts" << std::endl;
                    // for (auto [k, v]: testCounts2) {
                    //     std::cout << k << " " << v << " " <<  counts[k] << std::endl;
                    // }
                    // std::cout << std::endl;
                }

                // std::cout << "merge shift = " << shift << std::endl;
                // as_gpu.readN(test2.data(), n);
                // for (auto v: test2) {
                //     std::cout << v << " ";
                // }
                // std::cout << std::endl;

                get_counts_table.exec(gpu::WorkSize(workGroupSize, n), as_gpu, counts_gpu, shift, n);

                // std::cout << "counts_gpu shift = " << shift << std::endl;
                // counts_gpu.readN(test.data(), groupsCount * 4);
                // for (auto v: test) {
                //     std::cout << v << " ";
                // }
                // std::cout << std::endl;

                matrix_transpose.exec(gpu::WorkSize(workGroupSize, groupsCount), counts_gpu, counts_T_gpu, groupsCount);

                // std::cout << "counts_T_gpu shift = " << shift << std::endl;
                // counts_T_gpu.readN(test.data(), groupsCount * 4);
                // for (auto v: test) {
                //     std::cout << v << " ";
                // }
                // std::cout << std::endl;


                prefix_sum.exec(gpu::WorkSize(workGroupSize, groupsCount), counts_gpu, pref_gpu, groupsCount);

                // std::cout << "prefSum shift = " << shift << std::endl;
                // pref_gpu.readN(test.data(), groupsCount * 4);
                // for (auto v: test) {
                //     std::cout << v << " ";
                // }
                // std::cout << std::endl;

                uint size = n * 4;
                for (uint i = 1; i < size; i *= 2) {
					reduce.exec(gpu::WorkSize(workGroupSize, ((size + i - 1) / i) / 2), counts_T_gpu, i, size);
				}
				counts_T_gpu.readN(&res, 1, size - 1);
				for (uint i = size / 2; i >= 1; i /= 2) {
					scan.exec(gpu::WorkSize(workGroupSize, ((size + i - 1) / i) / 2), counts_T_gpu, i, int(i == size / 2), size);
				}

                move.exec(gpu::WorkSize(workGroupSize, size), counts_T_gpu, counts_pref_sum_gpu, size, res, size);

                // std::cout << "countspPrefSum shift = " << shift << std::endl;
                // counts_pref_sum_gpu.readN(test.data(), groupsCount * 4);
                // for (auto v: test) {
                //     std::cout << v << " ";
                // }
                // std::cout << std::endl;
                
                // radix.exec(gpu::WorkSize(workGroupSize, n), as_gpu, counts_pref_sum_gpu, pref_gpu, tmp, shift);
                radix.exec(gpu::WorkSize(workGroupSize, n), as_gpu, tmp, counts_pref_sum_gpu, pref_gpu, groupsCount, shift, n);

                std::swap(as_gpu, tmp);

                // std::cout << "radix" << std::endl;
                // as_gpu.readN(test2.data(), n);
                // for (auto v: test2) {
                //     testCounts[v]++;
                //     std::cout << v << " ";
                // }
                // std::cout << std::endl;

                // std::cout << "counts" << std::endl;
                // for (auto [k, v]: testCounts) {
                //     std::cout << k << " " << v << " " <<  counts[k] << std::endl;
                // }
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    // for (int i = 0; i < n; ++i) {
    //     std::cout << i << " " << bs[i] << " " << as[i] << " " << cpu_sorted[i] << std::endl;
    //     if (i > 10) break;
    //     // EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    // }

    for (int i = 0; i < n; ++i) {
        // std::cout << i << " " << bs[i] << " " << as[i] << " " << cpu_sorted[i] << std::endl;
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
