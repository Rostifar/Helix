#include "cuda.h"
#include "../CudaUtilities.cuh"
#define GAUSSIAN 4096
#define UNIFORM  24

namespace Helix {

template<typename F>
__global__ void generate(curandState *_state, F *bodies, F *limits, int offset, int typeOffset, int stride) {
	int idx 		  = blockDim.x * blockIdx.x + threadIdx.x;
	int j			  = 0;
	int typeIdx	      = idx + typeOffset;
	curandState state = _state[idx];

	for (int q = 0, p = 0; q < offset; q++) {
		if (q % stride == 0) {
			typeIdx += typeOffset;
			p 		+= 2;
			q++; 	continue;
		}
		switch((int)(bodies[typeIdx])) {
			case GAUSSIAN:
				F dl = (limits[p] - limits[p + 1] + 0.999999);
				bodies[idx + q] = ((typeid(F) == typeid(float)
						? curand_normal(&state) :  curand_normal_double(&state)) * dl) + limits[p + 1];
				break;
			case UNIFORM:
				bodies[idx + q] = ((typeid(F) == typeid(float)
						? curand_uniform(&state) : curand_uniform_double(&state)) * dl) + limits[p + 1];
				break;
		}
	}
}

template<typename F>
void generateParticles(F *limits, F *bodies, F *dBodies, dim3 blocks, dim3 threads, int offset, int typeOffset, int stride) {
	curandState *dStates;
	cudaMalloc(&dStates, allocationSize);
	cudaMemcpy(dStates, states, allocationSize, cudaMemcpyHostToDevice);

	F *dLimits = Helix::cudaAlloCopy(limits, sizeof(limits));
	dBodies    = Helix::cudaAlloCopy(bodies, sizeof(bodies));

	generate<F><<<blocks, threads>>>(dStates, dBodies, dLimits, offset, typeOffset, stride);
}
template void generateParticles<float>	(float *, float *, float *, dim3, dim3, int, int);
template void generateParticles<double> (double *, double *, double *, dim3, dim3, int, int);

}
