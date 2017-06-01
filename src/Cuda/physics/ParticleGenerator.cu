#include "cuda.h"
#include "../Types.cuh"
#define GAUSSIAN 4096
#define UNIFORM  24
#define NONE	 0
#define POISSON  96

namespace Helix {

template<typename F>
__device__ inline F getRandomSeed(F type, curandState *state) {
	switch ((int)type){
		case GAUSSIAN:
			return (typeid(F) == typeid(float) ? curand_normal(state) :  curand_normal_double(state));
		case UNIFORM:
			return (typeid(F) == typeid(float) ? curand_uniform(state) : curand_uniform_double(state));
	}
}

template<typename F>
__global__ void generateRandomParticles(F4<F> *particles, F4<F> *limits, curandState *states, int *strides, int offset) {
	int 		idx	  = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state = states[idx];

	for (int i = 0; i < offset; i++) {
		int distribution = (int) limits->w;
		F   dl 			 = limits->x - limits.y + 0.999999;

	}

	for (int i = 0, p = 0; i < sizeof(strides) / sizeof(strides[0]); i++, p += 3) {
		int distribution = (int) limits[p + 2];
		int elements     = strides[i];
		F   dl 			 = limits [p] - limits[p + 1] + 0.999999;

		if (distribution == NONE) idx += strides[i]; continue;

		for (int q = 0; q < elements; q++) {
			particles[idx] = (getRandomSeed<F>(distribution, state) * dl) + limits[p + 1];
			idx++;
		}
	}
}

template<typename F>
void distributionGeneration(F4<F> *_particles, F4<F> *_dParticles, GenerationLimits<F4<F>> *limits, int nParticles, dim3 *blocks, dim3 *threads, int offset, bool localCpy = false) {
	curandState *states = malloc(sizeof(curandState) * nParticles);
	curandState *dStates;

	cudaMalloc  (&dStates, sizeof(states));
	cudaMemcpy  (dStates, states, allocationSize, cudaMemcpyHostToDevice);

	F     *dLimits     = cudaAlloCopy<F>  (_limits, sizeof(limits->vec));
	int   *dStrides    = cudaAlloCopy<int>	  (_strides, sizeof(_strides));

	generateRandomParticles<F><<<*blocks, *threads>>>(_dParticles, dLimits, dStates, dStrides, offset);
	if (localCpy) {
		cudaMemcpy(_particles, _dParticles, size, cudaMemcpyDeviceToHost);
		cudaFree(_dParticles);
	}
	cudaFree(dLimits);
	cudaFree(dStrides);
	cudaFree(dStates);
	delete states;
}
template void distributionGeneration<float>  (float *, float *, float *, int, dim3 *, dim3 *, bool);
template void distributionGeneration<double> (double *, double *, double *, int, dim3 *, dim3 *, bool);

}
