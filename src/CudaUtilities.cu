#include "CudaUtilities.cuh"

namespace Helix {
	void *fixedCudaMalloc(size_t len) {
		void *p;
		if (cudaMalloc(&p, len) == cudaSuccess) return p;
		return 0;
	}

	template<typename FN>
	FN *cudaAlloCopy(FN *org, const size_t size) {
		void* mem = fixedCudaMalloc(size);
		FN *res = (FN *)mem;
		cudaMemcpy(res, org, size, cudaMemcpyHostToDevice);
		return res;
	}
	template float4 *cudaAlloCopy <float4>(float4 *, const size_t);
	template float3 *cudaAlloCopy <float3>(float3 *, const size_t);
	template double4 *cudaAlloCopy <double4>(double4 *, const size_t);
	template double3 *cudaAlloCopy <double3>(double3 *, const size_t);
}
