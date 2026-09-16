#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Copy data to GPU
            int *dev_odata, *dev_idata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            // Create device pts
            thrust::device_ptr<int> dev_thrust_idata(dev_idata);
            thrust::device_ptr<int> dev_thrust_odata(dev_odata);

            // Call thrusts's scan
            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);
            timer().endGpuTimer();
            
            // Copy to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Cleanup
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}
