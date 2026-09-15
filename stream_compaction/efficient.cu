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

        __global__ void kernDownSweep(int num_threads, int twoD, int twoDPlusOne, int *idata) {
            // Get thread index
            int k;
            if (!indexIsValid(num_threads, k)) return;
            k = k * twoDPlusOne;

            // Save left child
            int leftChild = idata[k + twoD - 1];
            
            // Set left child to this node's value
            idata[k + twoD - 1] = idata[k + twoDPlusOne - 1];

            // Set right child to old left child + this node
            idata[k + twoDPlusOne - 1] += leftChild;
        }

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

            // Pad with zeros 
            cudaMemset(dev_idata, 0, targetCnt * sizeof(int)); 
            checkCUDAError("cudaMemset dev_idata failed!");
            // Copy real data
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); 
            checkCUDAError("cudaMemcpy dev_idata failed!");
            
            // ----------------------------------
            // UP SWEEP
            int maxd = ilog2ceil(n) - 1;
            timer().startGpuTimer();
            // Traverse binary tree up
            for (int d=0; d<=maxd; d++) {
                int twoD = iTwoPow(d);
                int twoDPlusOne = iTwoPow(d+1);
                
                // Threads for this loop
                int n_d = targetCnt / twoDPlusOne;
                int fullBlocksPerGrid = divup(n_d, blockSize);
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(n_d, twoD, twoDPlusOne, dev_idata);
            }
            checkCUDAError("kernUpSweep failed!");

            // DEBUG
            // int* dbg = new int[targetCnt];
            // cudaMemcpy(dbg, dev_idata, targetCnt * sizeof(int), cudaMemcpyDeviceToHost);
            // printf("targetCnt = %d, root = %d\n", targetCnt, dbg[targetCnt - 1]);
        
            // ----------------------------------
            // DOWN SWEEP

            // Set root to zero
            cudaMemset(dev_idata + targetCnt - 1, 0, sizeof(int));
            // Traverse binary tree down
            for (int d=maxd; d>=0; d--) {
                int twoD = iTwoPow(d);
                int twoDPlusOne = iTwoPow(d+1);
                
                // Threads for this loop
                int n_d = targetCnt / twoDPlusOne;
                int fullBlocksPerGrid = divup(n_d, blockSize);
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n_d, twoD, twoDPlusOne, dev_idata);
            }
            timer().endGpuTimer();
            checkCUDAError("kernDownSweep failed!");

            // ----------------------------------
            // Copy memory from device->host
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // DEBUG
            // cudaMemcpy(dbg, dev_idata, targetCnt * sizeof(int), cudaMemcpyDeviceToHost);
            // printf("targetCnt = %d, root = %d\n", targetCnt, dbg[targetCnt - 1]);
            // delete[] dbg;

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
