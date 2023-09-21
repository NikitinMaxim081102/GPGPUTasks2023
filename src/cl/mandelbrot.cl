#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define threshold 256.0f
#define threshold2 65536.0f

__kernel void mandelbrot(__global       float* results,
                                  const uint   width, 
                                  const uint   height,
                                  const float  fromX, 
                                  const float  fromY,
                                  const float  sizeX, 
                                  const float  sizeY,
                                  const uint   iters, 
                                  const int    smoothing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const uint index_0 = get_global_id(0);
    const uint index_1 = get_global_id(1);

    if (index_0 >= width || index_1 >= height) return;

    float x0 = fromX + (index_0 + 0.5f) * sizeX / width;
    float y0 = fromY + (index_1 + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
    float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[index_1 * width + index_0] = result;
}