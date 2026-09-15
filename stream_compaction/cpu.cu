#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }
            timer().startCpuTimer();

            odata[0] = 0;
            for (int i=1; i<n; i++) {
                odata[i] = odata[i-1] + idata[i-1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int o_idx = 0;
            for (int i=0; i<n; i++) {
                if (idata[i] != 0) {
                    odata[o_idx] = idata[i];
                    o_idx++;
                }
            }

            timer().endCpuTimer();
            return o_idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *bools = new int[n];
            int *indices = new int[n];

            timer().startCpuTimer();

            // MAP TO BOOLEAN
            for (int i=0; i<n; i++) {
                bools[i] = idata[i] == 0 ? 0 : 1;
            }

            // SCAN 
            // (inlined: calling scan() would restart the already-running timer)
            indices[0] = 0;
            for (int i=1; i<n; i++) {
                indices[i] = indices[i-1] + bools[i-1];
            }

            // SCATTER
            int o_len = 0;
            for (int i=0; i<n; i++) {
                if (bools[i] == 1) {
                    o_len++;
                    odata[indices[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            
            delete[] bools;
            delete[] indices;

            return o_len;
        }
    }
}
