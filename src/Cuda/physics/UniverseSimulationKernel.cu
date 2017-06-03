#include <iostream>
#include <numeric>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "UniverseSimulation.h"
#include <stdio.h>

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
__device__ void cacheLocalBodies(F *interactingBodies, int start, int end) {
	for (; start <= end; start++) {
		interactingBodies[start] =
	}
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
void generateDistributedParticles( UniSimFmt<F> *_limits, UniParticle<F> *_particles, F *_dParticles, KernelDimensions *dims, int n, bool cpyLocal = false ) {
	curandState *states			= malloc( sizeof( curandState ) * n );
	curandState *dStates;
	F			*cudaLimits		= _limits->toCudaFmt();
	F			*dCudaLimits	= cudaAlloCopy<F>( cudaLimits, sizeof ( cudaLimits ) );
				*_dParticles	= cudaAlloCopy<F>(  )
}

template<typename F>
void startUniverseKernel(F epsilon, F dt, int n, int p, int epochs, UniLimitFmt<F> limits) {
	dim3 blocks  (n / p, 0, 0);
	dim3 threads (p, 0, 0);
	F *particles = malloc(sizeof(F) * n * UniParticle<F>::len);
	F *dParticles = cudaAlloCopy<F>(particles, sizeof(particles));

	distributionGeneration<F> (particles, dParticles, limits, params->particles, &blocks, &threads);

	for (int i = 0; i < params->epochs; i++) {
		simulateNaive<F, F3, F4><<<blocks, threads, sizeof(F4) * spec->partitions>>>(dBodies, dDynamics, dAccelerations, dt, epsilon, spec->particles);
		cudaMemcpy(bodies, dBodies, allocationSize,cudaMemcpyDeviceToHost); //implement callback to class
		cudaMemcpy(dynamics, dDynamics, allocationSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(accelerations, dAccelerations, allocationSize, cudaMemcpyDeviceToHost); //yo lance sucks
	}

	cudaFree(dParticles);
	delete particles;
}
template void beginUniSim <float> (UniverseParams<float> *, GenerationLimits<float> *);
template void beginUniSim <double>(UniverseParams<double> *, GenerationLimits<double> *);
}
