/*
 * CudaUtilities.cuh

 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */
#include "Types.cuh"

#ifndef CUDAUTILITIES_CUH_
#define CUDAUTILITIES_CUH_

namespace Helix {

typedef enum Platform {
	CPU,
	GPU,
	PolyGpu
};

void *fixedCudaMalloc(size_t len);

bool gpuSupported();

template<typename F>
F *cudaAlloCopy(F *org, const size_t size);

template<typename F>
static __device__ __inline__ __host__ F makeF4(F x, F y, F z, F w) {
	F4<F> t; t.x = x; t.y = y; t.z = z; t.w = w;
	return t;
}

template<typename F>
static __device__ __inline__ __host__ F makeF3(F x, F y, F z) {
	F3<F> t; t.x = x; t.y = y; t.z = z;
	return t;
}

template<typename F>
static __device__ __inline__ __host__ F makeF2(F x, F y) {
	F2<F> t; t.x = x; t.y = y;
	return t;
}

}
#endif /* CUDAUTILITIES_CUH_ */
