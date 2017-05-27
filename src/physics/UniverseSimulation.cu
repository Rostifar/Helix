#include <iostream>
#include <numeric>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "UniverseSimulation.cuh"
#include <stdio.h>
#include "../CudaUtilities.cuh"

__constant__ float G = 6.67300E-11;

template<typename F3, typename F4>
__device__ void calculatePartitionAcceleration(F4 body, F3 *acceleration) {
	for (int q = 0; q < blockDim.x; q++) {
		F3 tempA = calculateBodyAcceleration(body, interactingBodies[q]);
		acceleration->x += tempA.x;
		acceleration->y += tempA.y;
		acceleration->z += tempA.z;
	}
}

template<typename F, typename F3, typename F4>
__device__ F3 calculateBodyAcceleration(F4 bi, F4 bj) {
	F3 rij(bi.x - bj.x, bi.y - bj.y, bi.z - bj.z);
	F3 partialAcc (rij.x * bj.w, rij.y * bj.w, rij.z * bj.z);
	F smoothing = (rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + epsilonSquared);
	smoothing = smoothing * smoothing * smoothing;
	smoothing = sqrtf(smoothing);
	partialAcc.x /= smoothing;
	partialAcc.y /= smoothing;
	partialAcc.z /= smoothing;
	return partialAcc;
}

template<typename F, typename F3, typename F4>
__device__ float3 updateBodyVelocity(F3 a, F4 v, F dt) { //velocity verlet
	float3 newV;
	newV.x = v.x + 0.5 * a.x * dt;
	newV.y = v.y + 0.5 * a.y * dt;
	newV.z = v.z + 0.5 * a.z * dt;
	return newV;
}

template<typename F, typename F3, typename F4>
__device__ void updateBodyPosition(F3 velocity, F4 *r, float dt) {
	r->x += acceleration.x * dt;
	r->y += acceleration.y * dt;
	r->z += acceleration.z * dt;
}

template<typename F, typename F3, typename F4>
__global__ void simulateNaive(F4 *bodies, F4 *dynamics, F _dt, F _epsilon, int n_particles) {
	const int MAX_THREAD_COUNT = 1024;
	F dt = _dt;
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	int nParticles = n_particles;
	F4 body = bodies[particleId];
	F4 dynamic = dynamics[particleId];
	F3 velocity(dynamic.x, dynamic.y, dynamic.z);
	F3 r(body.x, body.y, body.z);
	F3 acceleration(0.0f, 0.0f, 0.0f);
	extern __shared__ float4 interactingBodies[];

	F3 vHalf = updateBodyVelocity(acceleration, velocity, dt, true);
	body.x += vHalf.x * dt;
	body.y += vHalf.y * dt;
	body.z += vHalf.z * dt;
	for (int i = 0; i < n_particles; i += blockDim.x) {
		interactingBodies[threadIdx.x] = bodies[threadIdx.x + i];
		__syncthreads();
		calculatePartitionAcceleration(&acceleration);
		__syncthreads();
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

template<typename F4>
__global__ void generateParticles(curandState *_state, float4 *bodies, float4 *dynamics, float4 *ranges) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	F4 body = bodies[idx];
	F4 dynamic = dynamics[idx];
	F4 range = ranges[idx];
	F4 range2 = ranges[idx + 1];
	curandState state = _state[idx];

	body.x = (curand_uniform(&state) * (range.x - range.y + 0.999999)) + range.y;
	body.y = (curand_uniform(&state) * (range.x - range.y + 0.999999)) + range.y;
	body.z = (curand_uniform(&state) * (range.x - range.y + 0.999999)) + range.y;
	dynamic.x = (curand_uniform(&state) * (range.z - range.w + 0.999999)) + range.w;
	dynamic.y = (curand_uniform(&state) * (range.z - range.w + 0.999999)) + range.w;
	dynamic.z = (curand_uniform(&state) * (range.z - range.w + 0.999999)) + range.w;
	body.w = (curand_uniform(&state) * (range2.x - range2.y + 0.999999)) + range2.y;
}

template<class F, class F3, class F4>
void beginSimulation(UniverseSimSpec<F> *spec, F4 *ranges) {
	size_t allocationSize = sizeof(F4) * spec->particles;
	F dt = spec->dt;
	F epsilon = spec->epsilon;
	F4 *bodies = (F4 *)malloc(allocationSize);
	F4 *dynamics = (F4 *)malloc(allocationSize);
	F4 *states = (F4 *)malloc(allocationSize);
	F3 *accelerations = (F3 *)malloc(allocationSize);
	F4 *generationRanges = (F4 *)malloc(allocationSize * 2); //[pos_h, pos_l, vel_h, vel_l]; [mass_h, mass_l, etc, etc]
	F4 *dBodies, *dDynamics, *dGenerationRanges;
	F3 *dAccelerations;
	dim3 blocks(spec->particles / spec->partitions, 0, 0);
	dim3 threads(spec->partitions, 0, 0);
	curandState *dStates;

	cudaMalloc(&dStates, allocationSize);
	cudaMemcpy(dStates, states, allocationSize, cudaMemcpyHostToDevice);
	cudaAlloCopy<F4>(bodies, dBodies, allocationSize);
	cudaAlloCopy<F4>(dynamics, dDynamics, allocationSize);
	cudaAlloCopy<F4>(ranges, dBodies, allocationSize * 2);
	cudaAlloCopy<F3>(accelerations, dAccelerations, allocationSize);
	generateParticles<F4><<<blocks, threads>>>(dStates, dBodies, dDynamics, dGenerationRanges);

	for (int i = 0; i < spec->epochs; i++) {
		simulateNaive<F, F3, F4><<<blocks, threads, sizeof(F4) * spec->partitions>>>(dBodies, dDynamics, dt, epsilon, spec->particles);
		cudaMemcpy(bodies, dBodies, cudaMemcpyDeviceToHost); //copy back to save to binary file
		cudaMemcpy(dynamics, dDynamics, cudaMemcpyDeviceToHost);
		cudaMemcpy(generationRanges, dGenerationRanges, cudaMemcpyDeviceToHost);
		cudaMemcpy(accelerations, dAccelerations, cudaMemcpyDeviceToHost); //yo lance sucks
	}
}
template void beginSimulation <float, float3, float4>(UniverseSimSpec<float> *, float4 *);
template void beginSimulation <double, double3, double4>(UniverseSimSpec<double> *, double4 *);
