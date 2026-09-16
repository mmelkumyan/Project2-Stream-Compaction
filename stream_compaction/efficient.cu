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

        void scanDeviceData(int n, int paddedN, int *dev_data) {
            // ---UP SWEEP---
            int maxd = ilog2ceil(n) - 1;
            
            // Traverse binary tree up
            for (int d=0; d<=maxd; d++) {
                int twoD = iTwoPow(d);
                int twoDPlusOne = iTwoPow(d+1);
                
                // Threads for this loop
                int n_d = paddedN / twoDPlusOne;
                int fullBlocksPerGrid = divup(n_d, blockSize);
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(n_d, twoD, twoDPlusOne, dev_data);
            }
            checkCUDAError("kernUpSweep failed!");

            // ---DOWN SWEEP---
            // Set root to zero
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
            // Traverse binary tree down
            for (int d=maxd; d>=0; d--) {
                int twoD = iTwoPow(d);
                int twoDPlusOne = iTwoPow(d+1);
                
                // Threads for this loop
                int n_d = paddedN / twoDPlusOne;
                int fullBlocksPerGrid = divup(n_d, blockSize);
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(n_d, twoD, twoDPlusOne, dev_data);
            }
            
            checkCUDAError("kernDownSweep failed!");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Pad to power of two
            int paddedN = iTwoPow(ilog2ceil(n));

            // Allocate device mem
            int *dev_data; 
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            // Pad with zeros 
            cudaMemset(dev_data, 0, paddedN * sizeof(int)); 
            checkCUDAError("cudaMemset dev_data failed!");
            // Copy real data
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice); 
            checkCUDAError("cudaMemcpy dev_data failed!");
            
            // Call helper scan
            timer().startGpuTimer();
            scanDeviceData(n, paddedN, dev_data);
            timer().endGpuTimer();

            // ----------------------------------
            // Copy memory from device->host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Cleanup
            cudaFree(dev_data);
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
            // Pad n
            int paddedN = iTwoPow(ilog2ceil(n));
            int fullBlocksPerGrid = divup(n, blockSize);

            // Allocate device mem
            int *dev_odata, *dev_idata, *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc bools failed!");
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            // Pad with zeros 
            cudaMemset(dev_indices, 0, paddedN * sizeof(int)); 
            checkCUDAError("cudaMemset dev_indices failed!");

            // Copy input
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); 
            checkCUDAError("cudaMemcpy dev_idata failed!");

            timer().startGpuTimer();
            // Get boolean map
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(
                n, dev_bools, dev_idata
            );

            // Copy bools -> indices array 
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice); 
            checkCUDAError("cudaMemcpy dev_indices failed!");
            
            // Get indices map
            scanDeviceData(n, paddedN, dev_indices);

            // Scatter to final array
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(
                n, dev_odata, dev_idata, dev_bools, dev_indices
            );
            timer().endGpuTimer();

            // Calc length of scattered array
            int finalBool, scatterLen;
            
            // Get the number of hits from indices array
            cudaMemcpy(&scatterLen,  dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            // Account for missing final index
            cudaMemcpy(&finalBool,  dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            scatterLen += finalBool;

            // Transfer output
            cudaMemcpy(odata, dev_odata, scatterLen * sizeof(int), cudaMemcpyDeviceToHost); 
            checkCUDAError("cudaMemcpy odata failed!");

            // Cleanup
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return scatterLen;
        }
    }
}
