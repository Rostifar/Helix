/*
 * CudaUtilities.cuh

 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */

#ifndef CUDAUTILITIES_CUH_
#define CUDAUTILITIES_CUH_

#include <cuda.h>

namespace Helix {

typedef struct KernelDimensions {
	dim3 blocks;
	dim3 threads;
};

void *fixedCudaMalloc(size_t len);

bool gpuSupported();

template<typename F>
F *cudaAlloCopy(F *org, const size_t size);

}
#endif /* CUDAUTILITIES_CUH_ */
