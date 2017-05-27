#include "CudaUtilities.cuh"

void *fixedCudaMalloc(size_t len) {
	void *p;
	if (cudaMalloc(&p, len) == cudaSuccess) return p;
	return 0;
}

template<typename FN>
void cudaAlloCopy(FN *org, const size_t size) {
	void* mem = fixedCudaMalloc(size);
	FN *res = (FN *)mem;
	cudaMemcpy(res, org, size, cudaMemcpyHostToDevice);
}
template void cudaAlloCopy <float4>(float4 *, float4 *, const size_t);
template void cudaAlloCopy <float3>(float3 *, float3 *, const size_t);
template void cudaAlloCopy <double4>(double4 *, double4 *, const size_t);
template void cudaAlloCopy <double3>(double3 *, double3 *, const size_t);
