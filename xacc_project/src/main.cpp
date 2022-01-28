/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

*/
#include <ap_int.h>
#include <iostream>
#include <algorithm>

#include "data_path.hpp"

#include <sys/time.h>
#include <new>
#include <cstdlib>

#include "logger.hpp"

#ifndef HLS_TEST
#include <xcl2.hpp>
#endif

#define M_PI 3.14159265

class ArgParser {
   public:
    ArgParser(int& argc, const char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

inline int tvdiff(struct timeval* tv0, struct timeval* tv1) {
    return (tv1->tv_sec - tv0->tv_sec) * 1000000 + (tv1->tv_usec - tv0->tv_usec);
}

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

//std::string binaryFile = xcl::find_binary_file(device_name, "vision_warp");
//cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

int main(int argc, char** argv) {
#ifndef HLS_TEST
    // cmd parser
    ArgParser parser(argc, (const char**)argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
    	std::cout << xclbin_path << "\n\r";
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif

    // set repeat time
    int nffts = 2;
#ifndef HLS_TEST
    std::string num_str;
    if (parser.getCmdOption("-n", num_str)) {
        try {
            nffts = std::stoi(num_str);
        } catch (...) {
            nffts = 1024;
        }
    }
#endif
    if (nffts > N_FFT) {
        std::cout << "Exceeding maximum number of FFT, reassigned number of FFT with " << N_FFT << std::endl;
        nffts = N_FFT;
    }

    std::cout << "nffts: " << nffts << std::endl;
    std::cout << "N_FFT: " << N_FFT << std::endl;
    std::cout << "FFT_LEN: " << FFT_LEN << std::endl;
    std::cout << "SSR: " << SSR << std::endl;

    ap_uint<512>* inData = aligned_alloc<ap_uint<512> >(FFT_LEN * nffts / SSR);
    ap_uint<512>* outData = aligned_alloc<ap_uint<512> >(FFT_LEN * nffts / SSR);
    // impulse as input
/*
    for (int n = 0; n < nffts; ++n) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
            if (t == 0){
                //float f = 1.0;
                //unsigned long x = *reinterpret_cast<unsigned long*>(&f);
                //inData[n * FFT_LEN / SSR + t] = x;
            	//inData[n * FFT_LEN / SSR + t].range(31 + 64 * t, 64 * t) = 1;
                std::cout << inData[n * FFT_LEN / SSR + t] << std::endl;

            }
            else{
                float f = 1.0;
                unsigned long x = *reinterpret_cast<unsigned long*>(&f);
                inData[n * FFT_LEN / SSR + t] = x;
                //inData[n * FFT_LEN / SSR + t].range(31 + 64 * t, 64 * t) = 1;
            	inData[n * FFT_LEN / SSR + t] = x;
                std::cout << inData[n * FFT_LEN / SSR + t] << std::endl;
            }
        }
    }
*/
/*
    float signal[8][8];
    int t=0;
    for (unsigned int i=0; i<8; i++){
        for (unsigned int j=0; j<8; j++){
        	signal[i][j] = (float)(10000*sin(2*M_PI*t/63))+10000;
        	std::cout << signal[i][j] << std::endl;
        	t++;
        }
    }
    for (unsigned int i=0; i<8; i++){
		inData[i].range(31, 0) = signal[i][0];
    	inData[i].range(95, 64) = signal[i][1];
    	inData[i].range(159, 128) = signal[i][2];
    	inData[i].range(223, 192) = signal[i][3];
    	inData[i].range(287, 256) = signal[i][4];
    	inData[i].range(351, 320) = signal[i][5];
    	inData[i].range(415, 384) = signal[i][6];
    	inData[i].range(479, 448) = signal[i][7];

    }
*/

    for (unsigned int i=0; i<8; i++){
    	if (i==0){
    		inData[i].range(31, 0) = 1;
    	} else {
    		inData[i].range(31, 0) = 1;
    	}
    	inData[i].range(95, 64) = 1;
    	inData[i].range(159, 128) = 1;
    	inData[i].range(223, 192) = 1;
    	inData[i].range(287, 256) = 1;
    	inData[i].range(351, 320) = 1;
    	inData[i].range(415, 384) = 1;
    	inData[i].range(479, 448) = 1;
    }


    std::cout << "Host buffer has been allocated and set.\n";

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
#ifndef HLS_TEST
    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl_int err;
    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    std::cout << xclbin_path << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);

    cl::Kernel kernel(program, "fft1DKernel", &err);
    logger.logCreateKernel(err);
    std::cout << "Kernel has been created.\n";

    cl_mem_ext_ptr_t mext_in, mext_out;
    mext_in = {XCL_MEM_DDR_BANK0, inData, 0};
    mext_out = {XCL_MEM_DDR_BANK0, outData, 0};

    cl::Buffer in_buff;
    in_buff = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                         (size_t)(sizeof(ap_uint<512>) * (FFT_LEN * nffts / SSR)), &mext_in);
    cl::Buffer out_buff;
    out_buff = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                          (size_t)(sizeof(ap_uint<512>) * (FFT_LEN * nffts / SSR)), &mext_out);
    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";

    q.finish();

    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);

    std::vector<std::vector<cl::Event> > write_events(1);
    write_events[0].resize(1);
    std::vector<std::vector<cl::Event> > kernel_events(1);
    kernel_events[0].resize(1);
    std::vector<std::vector<cl::Event> > read_events(1);
    read_events[0].resize(1);

    // write data to DDR
    std::vector<cl::Memory> ib;
    ib.push_back(in_buff);
    for (unsigned int i = 0; i < ib.size(); i++){
    	std::cout << ib[i].get();
    }
    q.enqueueMigrateMemObjects(ib, 0, nullptr, &write_events[0][0]);
    q.finish();
    std::cout << "H2D data transfer done.\n";

    // set args and enqueue kernel
    int j = 0;
    kernel.setArg(j++, in_buff);
    kernel.setArg(j++, out_buff);
    kernel.setArg(j++, nffts);
    q.enqueueTask(kernel, &write_events[0], &kernel_events[0][0]);
    q.finish();
    std::cout << "Kenrel execution done.\n";

    // read data from DDR
    std::vector<cl::Memory> ob;
    ob.push_back(out_buff);
    q.enqueueMigrateMemObjects(ob, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[0], &read_events[0][0]);
    q.finish();
    std::cout << "D2H data transfer done.\n";

    // wait all to finish
    gettimeofday(&end_time, 0);

    // profiling h2d, kernel, d2h times
    cl_ulong ts, te;
    write_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    write_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    float elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_H2D_MS, elapsed);
    kernel_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    kernel_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_KERNEL_MS, elapsed);
    read_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    read_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_D2H_MS, elapsed);

    // total execution time from CPU wall time
    std::cout << "Total execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
#else
    fftKernel<fftParams, IID, complex_wrapper<float> >(inData, outData, nffts);
#endif

    // check results
    int errs = 0;
    // step as output
    for (int n = 0; n < nffts; ++n) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
            for (int r = 0; r < SSR; r++) {
                if (outData[n * FFT_LEN / SSR + t].range(31 + 64 * r, 64 * r) != 1 ||
                    outData[n * FFT_LEN / SSR + t].range(63 + 64 * r, 32 + 64 * r) != 0) {
                    errs++;
                    std::cout << "Real = " << outData[n * FFT_LEN / SSR + t].range(31 + 64 * r, 64 * r)
                              << "    Imag = " << outData[n * FFT_LEN / SSR + t].range(63 + 64 * r, 32 + 64 * r)
                              << std::endl;
                }
            }
        }
    }
    int count=0;
    for (int n = 0; n < nffts; ++n) {
        for (int t = 0; t < FFT_LEN / SSR; ++t) {
        	for (int iter = 0; iter < 8; ++iter) {
        		count++;
				std::cout << "Sample[" << count <<"]: "<< "Real = " << outData[n * FFT_LEN / SSR + t].range(31 + 64 * iter, 64 * iter)
						  << "    Imag = " << outData[n * FFT_LEN / SSR + t].range(63 + 64 * iter, 32 + 64 * iter)
						  << std::endl;

        	}
        }
    }

    errs ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return errs;
}
