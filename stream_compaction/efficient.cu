#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int num_threads, int twoD, int twoDPlusOne, int *idata) {
            // Get thread index
            int k;
            if (!indexIsValid(num_threads, k)) return;
            k = k * twoDPlusOne;
            
            idata[k + twoDPlusOne - 1] += idata[k + twoD - 1];
        }

        // __global__ void kernDownSweep(int n, int *odata, const int *idata) {
        //     // Get thread index
        //     int k;
        //     if (!indexIsValid(n, k)) return;
            
        //     // TODO
        // }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Pad to power of two
            int targetCnt = iTwoPow(ilog2ceil(n));

            // Allocate device mem
            int *dev_idata; 
            cudaMalloc((void**)&dev_idata, targetCnt * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            // Copy memory from host->device
            //cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); // Copy real data
            //checkCUDAError("cudaMemcpy dev_idata failed!");
            //cudaMemset(dev_idata + n, 0, (targetCnt-n) * sizeof(int)); // Pad with zeros
            //checkCUDAError("cudaMemset dev_idata failed!");

            cudaMemset(dev_idata, 0, targetCnt * sizeof(int)); // Pad with zeros
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); // Copy real data
            checkCUDAError("cudaMemcpy dev_idata failed!");

            // DEBUG
            // cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            // ----------------------------------
            // UP SWEEP
            int maxd = ilog2ceil(n) - 1;
            timer().startGpuTimer();
            for (int d=0; d<=maxd; d++) {
                int twoD = iTwoPow(d);
                int twoDPlusOne = iTwoPow(d+1);
                
                // Threads for this loop
                int n_d = targetCnt / twoDPlusOne;
                int fullBlocksPerGrid = divup(n_d, blockSize);
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(n_d, twoD, twoDPlusOne, dev_idata);
            }
            timer().endGpuTimer();
            checkCUDAError("kernScan failed!");
        
            // DOWN SWEEP
            // ----------------------------------
            // TODO...


            // ----------------------------------
            // Copy memory from device->host
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);




            // Cleanup
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
