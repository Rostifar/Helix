#include <iostream>
#include <numeric>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "UniverseSimulation.h"
#include <stdio.h>
#include "GenerationUtils.cu"

namespace Helix {
__constant__ float G = 6.67300E-11;

/*
 * http://www.scholarpedia.org/article/N-body_simulations_(gravitational)
 */

template<typename F, typename F3, typename F4>
__device__ F3 calculateBodyAcceleration(F4 bi, F4 bj, F epsilonSquared) {
	F3 rij = {bi.x - bj.x, bi.y - bj.y, bi.z - bj.z};
	F3 partialAcc = {rij.x * bj.w, rij.y * bj.w, rij.z * bj.z};
	F smoothing = (rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + epsilonSquared);
	smoothing = smoothing * smoothing * smoothing;
	smoothing = sqrtf(smoothing);
	partialAcc.x /= smoothing;
	partialAcc.y /= smoothing;
	partialAcc.z /= smoothing;
	return partialAcc;
}

template<typename F, typename F3, typename F4>
__device__ void calculatePartitionAcceleration(F4 body, F3 *acceleration, F4 *sharedParticles, F epsilon) {
	for (int q = 0; q < blockDim.x; q++) {
		F3 tempA = calculateBodyAcceleration<F, F3, F4>(body, sharedParticles[q], epsilon * epsilon);
		acceleration->x += tempA.x;
		acceleration->y += tempA.y;
		acceleration->z += tempA.z;
	}
}

template<typename F>
__device__ inline void updateBodyVelocity(F *body, const F dt) { //velocity verlet (https://en.wikipedia.org/wiki/Verlet_integration)
	body[3] += 0.5 * body[6] * dt;
	body[4] += 0.5 * body[7] * dt;
	body[5] += 0.5 * body[8] * dt;
}

template<typename F>
__device__ inline void updateBodyPosition(F *body, const F dt) {
	body[0] += body[3] * dt;
	body[1] += body[4] * dt;
	body[2] += body[5] * dt;
}

template<typename F>
__global__ void simulateNaive(F *particles, F time, const F epsilon, const F dt, const int offset) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int particleId = globalId * offset;
	const int localIdx = threadIdx.x * offset;
	const int positionDim = 3;
	F body[offset];
	__shared__ F interactingBodies[]; //only needs bodies and masses

	for (int i = particleId, q = 0, p = localIdx; i < (particleId + offset); i++, q++, p++) {
		if (q > 3) interactingBodies[p] = particles[i];
		body[q] = particles[i];
	}
	__syncthreads();
	updateBodyVelocity<F>(body, dt); //half-step
	updateBodyPosition<F>(particle, dt);

	for (int i = 0, localIdx = threadIdx.x * offset; i < n_particles; i += blockDim.x) { //including self reaction
		cacheLocalBodies(interactingBodies, localIdx, localIdx + offset);
		__syncthreads();
		calculatePartitionAcceleration<F, F3, F4>(body, &acceleration, interactingBodies, _epsilon); //pass empty custon struct to array
		__syncthreads();
		localIdx += i;
	}
	acceleration.x *= G;
	acceleration.y *= G;
	acceleration.z *= G;
	velocity = updateBodyVelocity(acceleration, vHalf, dt);
	dynamic.x = velocity.x;
	dynamic.y = velocity.y;
	dynamic.z = velocity.z;

	printf("Particle: %i \n", particleId);
	printf("Acceleration: %f %f %f \n", acceleration.x, acceleration.y, acceleration.z);
	printf("Velocity: %f %f %f \n", velocity.x, velocity.y, velocity.z);
	printf("Position: %f %f %f \n", body.x, body.y, body.z);
	printf("\n");

	bodies[particleId] = body;
	dynamics[particleId] = dynamic;
}

template<typename F>
__global__ void distributionGeneration(F *particles, F* limits, curandState *states) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state = states[idx];
	const int genTypeIdx = 8;
	const int components[4] = {3, 3, 3, 1};
	const int limitStride = 2;
	const int limitSize = 12;

	__shared__ F sharedLimits[limitSize];

	if (threadIdx.x < limitSize) sharedLimits[threadIdx.x] = limits[threadIdx.x];

	for (int i = 0, q = 0; limitStride < genTypeIdx; i += limitStride, genTypeIdx++, q++) {
		int genType = (int) sharedLimits[genType];
		if (genType != NONE) {
			for (int p = 0; p < components[q]; p++, idx++) {
				particles[idx] = (getRandomSeed<F>(genType, state) * dl) +  sharedLimits[i + 1];
			}
		}
	}
}

template<typename F>
void generateDistributedParticles(UniSimFmt<F> *_limits, UniParticle<F> *_particles, F *_dParticles, KernelDimensions *dims, int n, bool cpyLocal = false) {
	curandState *states = malloc(sizeof(curandState) * n);
	curandState *dStates;
	cudaMalloc  (&dStates, sizeof(states));
	cudaMemcpy  (dStates, states, allocationSize, cudaMemcpyHostToDevice);

	F *cudaLimits = _limits->toCudaFmt();
	F *dCudaLimits = cudaAlloCopy<F>(cudaLimits, sizeof(cudaLimits));
	F *particles = new F[n * UniParticle<F>::len];
	UniParticle<F>::massCudaFmt(particles, _particles, n);
	_dParticles = cudaAlloCopy<F>(particles, sizeof(particles));

	distributionGeneration<<<dims->blocks, dims->threads>>>(_dParticles, dCudaLimits, dStates);

	if (cpyLocal) {
		F *arr = malloc(sizeof(F) * n * UniParticle<F>::len);
		cudaMemcpy(arr, _dParticles, sizeof(_dParticles), cudaMemcpyDeviceToHost);
		UniParticle<F>::massHostFmt(arr, _particles, n);
		delete arr;
		cudaFree(_dParticles); //otherwise can be used again for particle calculations.
	}

	delete states;
	delete cudaLimits;
	delete particles;
	cudaFree(dCudaLimits);
	cudaFree(dStates);
}
template void generateDistributedParticles<float>(UniSimFmt<float> *, UniParticle<float> *, float *, KernelDimensions *, int, bool);
template void generateDistributedParticles<double>(UniSimFmt<double> *, UniParticle<double> *, double *, KernelDimensions *, int, bool);

template<typename F>
void startUniverseKernel(UniSimParams<F> *params, F *dParticles, KernelDimensions *dims, const int epochs) { //preallocated particles
	for (int i = 0; i < epochs; i++) {
		simulateNaive<F>()
	}
}

}

/*
template<typename F>
	dim3 blocks  (n / p, 0, 0);
	dim3 threads (p, 0, 0);
	F *particles = malloc(sizeof(F) * n * UniParticle<F>::len);
	F *dParticles = cudaAlloCopy<F>(particles, sizeof(particles));
	distributionGeneration<F> (particles, dParticles, limits, params->particles, &blocks, &threads);

	for (int i = 0; i < params->epochs; i++) {
		simulateNaive<F, F3, F4><<<blocks, threads, sizeof(F4) * spec->partitions>>>(dBodies, dDynamics, dAccelerations, dt, epsilon, spec->particles);
		cudaMemcpy(bodies, dBodies, allocationSize, cudaMemcpyDeviceToHost); //implement callback to class
		cudaMemcpy(dynamics, dDynamics, allocationSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(accelerations, dAccelerations, allocationSize, cudaMemcpyDeviceToHost); //yo lance sucks
	}

	cudaFree(dParticles);
	delete particles;
}
template void beginUniSim <float> (UniverseParams<float> *, GenerationLimits<float> *);
template void beginUniSim <double>(UniverseParams<double> *, GenerationLimits<double> *);
}
*/
