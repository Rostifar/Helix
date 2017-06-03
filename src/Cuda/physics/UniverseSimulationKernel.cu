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
__device__ inline void updateBodyVelocity(F *particle, F dt) { //velocity verlet
	particle[3] += 0.5 * particle[6] * dt;
	particle[4] += 0.5 * particle[7] * dt;
	particle[5] += 0.5 * particle[8] * dt;
}

template<typename F>
__device__ inline void updateBodyPosition(F *particle, F dt) {
	particle[0] += particle[3] * dt;
	particle[1] += particle[4] * dt;
	particle[2] += particle[5] * dt;
}

template<typename F>
__global__ void simulateNaive(F *particles, int offset, F _dt, F _epsilon, int n_particles) {
	int        particleId = blockDim.x * blockIdx.x + threadIdx.x;
	int 	   idx 	      = particleId * offset;
	F 		   particle[10];
	__shared__ F interactingBodies[];

	for (int i = 0; i < offset; i++) particle[i] = particles[idx + i];

	updateBodyVelocity<F>(particle, dt);
	updateBodyPosition<F>(particle, dt);

	for (int i = 0, localIdx = threadIdx.x * offset; i < n_particles; i += blockDim.x) {
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
