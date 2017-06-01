#include <typeinfo>
#include "CudaUtilities.cuh"
#include "cuda.h"

namespace Helix {

void *fixedCudaMalloc(size_t len) {
	void *p;
	if (cudaMalloc(&p, len) == cudaSuccess) return p;
	return 0;
}

bool gpuSupported() {
	int devices = 0;
	cudaError_t err = cudaGetDeviceCount(&devices);
	return devices > 0 && err == cudaSuccess;
}

template<typename F>
F *cudaAlloCopy(F *org, const size_t size) {
	void *mem = fixedCudaMalloc(size);
	F 	 *res = (F *)mem;
	cudaMemcpy(res, org, size, cudaMemcpyHostToDevice);
	return res;
}
template float *cudaAlloCopy <float> (float *, const size_t);
template double *cudaAlloCopy <double> (double *, const size_t);
}
