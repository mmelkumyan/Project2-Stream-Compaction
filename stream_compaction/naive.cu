#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"



namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanD(int n, int offset, int *odata, const int *idata) {
            // Get thread index
            int k;
            if (!indexIsValid(n, k)) return;
            
            // Make sure we dont get a negative index
            if (k >= offset) {
                // Prev + current idx
                odata[k] = idata[k - offset] + idata[k];
            } else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         * Reference: GPU Gems 3 https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
         */
        void scan(int n, int *odata, const int *idata) {
            int fullBlocksPerGrid = divup(n, blockSize);

            int *dev_tempA, *dev_tempB;            

            // Allocate device memory for each array
            cudaMalloc((void**)&dev_tempA, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_tempA failed!");
            cudaMalloc((void**)&dev_tempB, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_tempB failed!");

            // Copy memory from host->device
            // Only seed A, B will be overwritten!
            cudaMemcpy(dev_tempA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Invoke scan threads
            int maxd = ilog2ceil(n);
            timer().startGpuTimer();
            for (int d=1; d<=maxd; d++) {

                int offset = iTwoPow(d-1);
                kernScanD<<<fullBlocksPerGrid, blockSize>>>(n, offset, dev_tempB, dev_tempA);

                // Swap A <-> B
                // This iteration output = next iteration input
                // This iteration input -> overwritten for next iteration output
                std::swap(dev_tempA, dev_tempB);
            }
            timer().endGpuTimer();
            checkCUDAError("kernScan failed!");
            
            // Copy memory from device->host
            // Move to the right 1 index and replace index 0 w/ 0 (to make it exclusive)
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_tempA, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);

            // Cleanup
            cudaFree(dev_tempA);
            cudaFree(dev_tempB);
        }
    }
}
