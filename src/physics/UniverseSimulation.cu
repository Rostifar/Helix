#include <iostream>
#include <numeric>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "UniverseSimulation.cuh"

__constant__ float G = 6.67300E-11;
const int MAX_THREAD_SIZE = 1024;

template<class F3>
__device__ void calculatePartitionAcceleration(F3 *acceleration) {
	for (int q = 0; q < blockDim.x; q++) {
		F3 tempA = calculateBodyAcceleration(body, interactingBodies[q]);
		acceleration->x += tempA.x;
		acceleration->y += tempA.y;
		acceleration->z += tempA.z;
	}
}

template<class F, class F3, class F4>
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

template<class F, class F3, class F4>
__device__ float3 updateBodyVelocity(F3 a, F4 v, F dt) { //velocity verlet
	float3 newV;
	newV.x = v.x + 0.5 * a.x * dt;
	newV.y = v.y + 0.5 * a.y * dt;
	newV.z = v.z + 0.5 * a.z * dt;
	return newV;
}

template<class F, class F3, class F4>
__device__ void updateBodyPosition(F3 velocity, F4 *r, float dt) {
	r->x += acceleration.x * dt;
	r->y += acceleration.y * dt;
	r->z += acceleration.z * dt;
}

template<class F, class F3, class F4>
__global__ void simulateNaive(F4 *bodies, F4 *dynamics, F _dt, int n_particles) {
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

	bodies[particleId] = body;
	dynamics[particleId] = dynamic;
}

template<class F4>
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
void beginSimulation(int numberOfParticles, int partitions, int epochs, F _dt, F _epsilon) { //add ability to serialize from past renders.
	size_t allocationSize = sizeof(F4) * numberOfParticles;
	size_t rangeAllcSize = sizeof(F4) * numberOfParticles * 2;
	F dt = _dt;
	F epsilon = _epsilon;
	F3 *bodies = malloc(allocationSize);
	F4 *dynamics = malloc(allocationSize);
	F4 *generationRanges = malloc(rangeAllcSize); //[pos_h, pos_l, vel_h, vel_l]; [mass_h, mass_l, etc, etc]
	F3 *accelerations = malloc(allocationSize);
	F4 *dBodies;
	F4 *dDynamics;
	F4 *dGenerationRanges;
	F4 *dAccelerations;
	dim3 blocks(numberOfParticles / partitions, 0, 0);
	dim3 threads(partitions, 0, 0);
	curandState *dState;

	cudaMalloc(&dState, blocks.x * threads.x);
	cudaMalloc((void**) &dBodies, allocationSize);
	cudaMalloc((void**) &dDynamics, allocationSize);
	cudaMalloc((void**) &dGenerationRanges, rangeAllcSize);
	cudaMalloc((void**) &dAccelerations, allocationSize);

	cudaMemcpy(dBodies, bodies, cudaMemcpyHostToDevice);
	cudaMemcpy(dDynamics, dynamics, cudaMemcpyHostToDevice);
	cudaMemcpy(dGenerationRanges, generationRanges, cudaMemcpyHostToDevice);
	cudaMemcpy(dAccelerations, accelerations, cudaMemcpyHostToDevice);
	generateParticles<<<blocks, threads>>>(dBodies, dDynamics, dGenerationRanges);

	for (int i = 0; i < epochs; i++) {
		simulateNaive<<<blocks, threads, sizeof(F4) * partitions>>>(dBodies, dDynamics, numberOfParticles, dt, epochs);
		cudaMemcpy(bodies, dBodies, cudaMemcpyDeviceToHost); //copy back to save to binary file
		cudaMemcpy(dyanmics, dDynamics, cudaMemcpyDeviceToHost);
		cudaMemcpy(generationRanges, dGenerationRanges, cudaMemcpyDeviceToHost);
		cudaMemcpy(accelerations, dAccelerations, Ranges, cudaMemcpyDeviceToHost); //yo lance sucks
	}
}

template<class F, class F3, class F4>
void resumeSimulation() {

}
