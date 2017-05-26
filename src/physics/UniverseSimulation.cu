#include <iostream>
#include <numeric>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"

__constant__ float epsilonSquared = 0.2;
__constant__ float G = 6.67300E-11;
__device__ float globalDt;

const int MAX_THREAD_SIZE = 1024;

__device__ int getGlobalId() {
	return blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getParticleId() {
	return blockIdx.x * blockDim.y + threadIdx.y;
}

inline __device__ float3 segmentFloat4(float4 f4) {
	return make_float3(f4.x, f4.y, f4.z);
}

__device__ void calculatePartitionAcceleration(float3 *acceleration) {
	for (int q = 0; q < blockDim.x; q++) {
		float3 tempA = calculateBodyAcceleration(body, interactingBodies[q]);
		acceleration->x += tempA.x;
		acceleration->y += tempA.y;
		acceleration->z += tempA.z;
	}
}

__device__ float3 calculateBodyAcceleration(float4 bi, float4 bj) {
	float3 rij(bi.x - bj.x, bi.y - bj.y, bi.z - bj.z);
	float3 partialAcc (rij.x * bj.w, rij.y * bj.w, rij.z * bj.z);
	float smoothing = (rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + epsilonSquared);
	smoothing = smoothing * smoothing * smoothing;
	smoothing = sqrtf(smoothing);
	partialAcc.x /= smoothing;
	partialAcc.y /= smoothing;
	partialAcc.z /= smoothing;
	return partialAcc;
}

__device__ float3 updateBodyVelocity(float3 a, float4 v, float dt) { //velocity verlet
	float3 newV;
	newV.x = v.x + 0.5 * a.x * dt;
	newV.y = v.y + 0.5 * a.y * dt;
	newV.z = v.z + 0.5 * a.z * dt;
	return newV;
}

__device__ void updateBodyPosition(float3 velocity, float4 *r, float dt) {
	r->x += acceleration.x * dt;
	r->y += acceleration.y * dt;
	r->z += acceleration.z * dt;
}

__global__ void simulateNaive(float4 *bodies, float4 *dynamics, int n_particles) {
	const int MAX_THREAD_COUNT = 1024;
	const float dt = 0.00001;
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	int nParticles = n_particles;
	float4 body = bodies[particleId];
	float4 dynamic = dynamics[particleId];
	float3 velocity(dynamic.x, dynamic.y, dynamic.z);
	float3 r(body.x, body.y, body.z);
	float3 acceleration(0.0f, 0.0f, 0.0f);
	extern __shared__ float4 interactingBodies[];

	float3 vHalf = updateBodyVelocity(acceleration, velocity, dt, true);
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

__global__ void generateParticles(float4 *bodies, float4 *dynamics, float4 *ranges) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	float4 body = bodies[idx];
	float4 dynamic = dynamics[idx];
	float4 range = ranges[idx];
	float4 range2 = ranges[idx];


}

void beginUniverseSimulation(int numberOfParticles, int partitions, float dt, int epochs) { //add ability to serialize from past renders.
	size_t allocationSize = sizeof(float4) * numberOfParticles;
	size_t rangeAllcSize = sizeof(float4) * numberOfParticles * 2;
	float4 *bodies = malloc(allocationSize);
	float4 *dynamics = malloc(allocationSize);
	float4 *generationRanges = malloc(rangeAllcSize); //[pos_l, pos_h, vel_l, vel_h]; [mass, etc, etc, etc]
	float3 *accelerations = malloc(allocationSize);
	float4 *dBodies;
	float4 *dDynamics;
	float4 *dGenerationRanges;
	float4 *dAccelerations;
	dim3 blocks(numberOfParticles / partitions, 0, 0);
	dim3 threads(partitions, 0, 0);

	curandState *dState;

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
		simulateNaive<<<blocks, threads, sizeof(float4) * partitions>>>(dBodies, dDynamics, numberOfParticles, dt, epochs);
		cudaMemcpy(bodies, dBodies, cudaMemcpyDeviceToHost); //copy back to save to binary file
		cudaMemcpy(dyanmics, dDynamics, cudaMemcpyDeviceToHost);
		cudaMemcpy(generationRanges, dGenerationRanges, cudaMemcpyDeviceToHost);
		cudaMemcpy(accelerations, dAccelerations, Ranges, cudaMemcpyDeviceToHost); //yo lance sucks
	}
}

void resumeUniverseSimulation() {

}
