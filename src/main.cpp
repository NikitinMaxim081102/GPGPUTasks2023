#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

struct Device {
    cl_platform_id platform;
    cl_device_id device;
};

Device getDevice() {
    Device bestDevice;
    cl_uint platformsCount = 0;
    cl_platform_id lastPlatform;
    cl_device_id lastDevice;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];

        // size_t platformNameSize = 0;
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // std::vector<unsigned char> platformName(platformNameSize, 0);
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), NULL));
        // std::cout << "    Platform name: " << platformName.data() << std::endl;

        // size_t platformVendorSize = 0;
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        // std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), NULL));
        // std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;


        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &devicesCount));
        // std::cout << "Number of OpenCL platforms: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, devicesCount, devices.data(), nullptr));
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {

            // std::cout << "Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL));
            // std::string deviceTypeString = "";
            if (deviceType == CL_DEVICE_TYPE_GPU) {
                bestDevice.platform = platform;
                bestDevice.device = device;
                return bestDevice;
                // deviceTypeString += "CPU";
            }
            else {
                lastPlatform = platform;
                lastDevice = device;
            }
            // std::cout << "    " << "    " << "Type: " << " " << deviceTypeString << std::endl;
        }

        std::cout << std::endl;
    }
    bestDevice.platform = lastPlatform;
    bestDevice.device = lastDevice;
    return bestDevice;
}


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    Device device = getDevice();

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device.device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    cl_command_queue queue = clCreateCommandQueue(context, device.device, 0, &err);
    OCL_SAFE_CALL(err);

    // unsigned int n = 1000 * 1000;
    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    
    cl_mem buffA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*as.size(), nullptr, &err);
    OCL_SAFE_CALL(err);

    cl_mem buffB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*bs.size(), nullptr, &err);
    OCL_SAFE_CALL(err);

    cl_mem buffC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*cs.size(), nullptr, &err);
    OCL_SAFE_CALL(err);


    clEnqueueWriteBuffer(queue, buffA, CL_TRUE, 0, sizeof(float)*as.size(), as.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, buffB, CL_TRUE, 0, sizeof(float)*bs.size(), bs.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, buffC, CL_TRUE, 0, sizeof(float)*cs.size(), cs.data(), 0, nullptr, nullptr);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    const char* src_text = kernel_sources.c_str();
    size_t src_length = kernel_sources.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_text, &src_length, &err);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    OCL_SAFE_CALL(clBuildProgram(program, 1, &device.device, nullptr, nullptr, nullptr));

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);

    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    if (log_size > 1) {
       std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    const char kernel_name[] = "aplusb";
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    OCL_SAFE_CALL(err);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &buffA);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &buffB);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &buffC);
        clSetKernelArg(kernel, i++, sizeof(size_t), &n);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event kernel_run;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, &kernel_run));
            OCL_SAFE_CALL(clWaitForEvents(1, &kernel_run));
            clReleaseEvent(kernel_run);
            // clEnqueueNDRangeKernel...
            // clWaitForEvents...
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / (t.lapAvg() * 1e9) << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3*n*sizeof(float) / (1024*1024*1024 * t.lapAvg()) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event read_buffer;
            // clEnqueueReadBuffer...
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, buffC, CL_TRUE, 0, n, cs.data(), 0, nullptr, &read_buffer));
            OCL_SAFE_CALL(clWaitForEvents(1, &read_buffer));
            clReleaseEvent(read_buffer);
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n*sizeof(float) / (1024*1024*1024 * t.lapAvg()) << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
       if (cs[i] != as[i] + bs[i]) {
            std::cout << "i = " << i << std::endl;
            std::cout << "a = " << as[i] << std::endl;
            std::cout << "b = " << bs[i] << std::endl;
            std::cout << "c = " << cs[i] << std::endl;
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }



    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffA);
    clReleaseMemObject(buffB);
    clReleaseMemObject(buffC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}