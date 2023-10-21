#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, const T &i, std::string message, std::string filename, int line)
{
	if (a != b) {
		std::cerr << message << " But at " << i << " " << a << " != " << b << ", " << filename << ":" << line << std::endl;
		throw std::runtime_error(message);
	}
}

#define EXPECT_THE_SAME(a, b, i, message) raiseFail(a, b, i, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
	int benchmarkingIters = 10;
	unsigned int max_n = (1 << 24);
	gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

	uint workGroupSize = 16;
	gpu::gpu_mem_32u as_gpu;
	ocl::Kernel reduce(prefix_sum_kernel, prefix_sum_kernel_length, "reduce");
	ocl::Kernel scan(prefix_sum_kernel, prefix_sum_kernel_length, "scan");
	reduce.compile();
	scan.compile();

	for (unsigned int n = 4096; n <= max_n; n *= 4) {
	// for (unsigned int n = 65536; n <= max_n; n *= 4) {
		std::cout << "______________________________________________" << std::endl;
		unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
		std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

		std::vector<unsigned int> as(n, 0);
		FastRandom r(n);
		for (int i = 0; i < n; ++i) {
			as[i] = r.next(0, values_range);
		}

		std::vector<unsigned int> bs(n, 0);
		{
			for (int i = 0; i < n; ++i) {
				bs[i] = as[i];
				if (i) {
					bs[i] += bs[i-1];
				}
			}
		}
		// for (uint i = 0; i < 3; ++i) {
		// 	std::cout << as[i] << " " << bs[i] << " " << std::endl;
		// }
		const std::vector<unsigned int> reference_result = bs;

		{
			{
				std::vector<unsigned int> result(n);
				for (uint i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				for (uint i = 0; i < n; ++i) {
					EXPECT_THE_SAME(reference_result[i], result[i], i, "CPU result should be consistent!");
				}
			}

			std::vector<unsigned int> result(n);
			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				t.nextLap();
			}
			std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}

		{
			as_gpu.resizeN(n);
			std::vector<unsigned int> result(n);
			uint res;
			timer t;
			// for (uint i = 0; i < 3; ++i) {
			// 	std::cout << as[i] << " " << bs[i] << " " << std::endl;
			// }
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				const std::vector<unsigned int> vals(as);
				as_gpu.writeN(vals.data(), n);

				t.restart();

				for (int i = 1; i < n; i *= 2) {
					reduce.exec(gpu::WorkSize(workGroupSize, (n + i - 1) / i), as_gpu, i);
				}
				as_gpu.readN(&res, 1, n - 1);
				for (int i = n / 2; i >= 1; i /= 2) {
					scan.exec(gpu::WorkSize(workGroupSize, (n + i - 1) / i), as_gpu, i, i == n / 2, n);
				}

				t.nextLap();
			}
			std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

			as_gpu.readN(result.data(), n - 1, 1);
			result[n-1] = res;
			// std::cout << res << " RESULT" << std::endl;
			// if (n == 65536) {
			// for (int i = 0; i < 3; ++i) {
			// 	std::cout << "as= " << as[i] << " " << bs[i] << " " << result[i] << " " << reference_result[i] << " " << i << std::endl;
			// }
			// }
			std::cout << std::endl;
			for (uint i = 0; i < n; ++i) {
				// std::cout << as[i] << " " << bs[i] << " " << i << std::endl;
				EXPECT_THE_SAME(result[i], bs[i], i, "GPU results should be equal to CPU results!");
			}
		}
	}
}
