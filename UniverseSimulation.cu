#include <iostream>
#include <numeric>
#include <math.h>

/*
 * offset
 * ------
 * [position<3>, mass], [velocity<3>, intensity]
 *
 * */

__constant__ float epsilonSquared = 0.2;
__constant__ float G = 6.67300E-11;
__device__ float globalDt;

const int MAX_THREAD_SIZE = 1024;

__global__ void generateParticles(float4 particle) {

}

__device__ int getGlobalId() {
	return blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getParticleId() {
	return blockIdx.x * blockDim.y + threadIdx.y;
}

inline __device__ float3 segmentFloat4(float4 f4) {
	return make_float3(f4.x, f4.y, f4.z);
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
	float3 newV = make_float3(0.0f, 0.0f, 0.0f);
	newV.x += a.
}

__device__ void updateBodyPosition(float3 velocity, float4 *r, float dt) {
	r->x += acceleration.x * dt;
	r->y += acceleration.y * dt;
	r->z += acceleration.z * dt;
}

/*
 * kernel format:
 * -------------
 *
 * THREAD:
 *  i = 1024/p;
 *
 * [1, 2, 3, 4, 5, ..., i]
 * 1,
 * 2,
 * 3,
 * 4,
 * 5,
 * .
 * .
 * .,
 * p
 * [					  ]
 *
 * specifications:
 * # Each row represents a single particle.
 * # At each time a single particle-row may process w elements in a shared memory body array.
 * # After each particle-particle reaction is calculated, the thread block "slides" to calculate the next i reactions until
 * (N / i) reactions are calculated.
 *
 * BLOCK:
 * w = N/p;
 *
 * [1, 2, 3, 4, 5, ..., w]
 *
 * input format:
 * ------------
 * #bodies => [position<3>, mass<1>]
 * #dynamics => [velocity<3>, 0] ; ~0 will be replaced by future value~
 *
 * */

__global__ void simulateNaive(float4 *bodies, float3 *dynamics, int n_particles, float _dt, int epochs) {
	const int MAX_THREAD_COUNT = 1024;
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	int nParticles = n_particles;
	float dt = _dt;
	float4 body = bodies[particleId];
	float4 dynamic = dynamics[particleId];
	float3 velocity(dynamic.x, dynamic.y, dynamic.z);
	float3 r(body.x, body.y, body.z);
	float3 acceleration(0.0f, 0.0f, 0.0f);
	extern __shared__ float4 interactingBodies[];

	for (int i = 0; i < n_particles; i += blockDim.x) {
		interactingBodies[threadIdx.x] = bodies[i];
		__syncthreads();
		for (int q = 0; q < blockDim.x; q++) {
			float3 tempA = calculateBodyAcceleration(body, interactingBodies[q]);
			acceleration.x += tempA.x;
			acceleration.y += tempA.y;
			acceleration.z += tempA.z;
		}
		acceleration *= G;
		updateBodyVelocity(acceleration, &velocity, dt);
		updateBodyPosition()
	}








	int filterSteps = nParticles / w;
	int filterIdx = threadIdx.x + (blockDim.x * threadIdx.y);

	float4 mainBody = bodies[particle_id];
	float4 dynamic = dynamics[paricle_id];
	float3 velocity = make_float3(dynamic.x, dynamic.y, dynamic.z);
	float3 acc;


	for (int q = 0; q < epochs; epochs++) {
		for (int j = 0; i < filterSteps; j++) {
			currentBodies[filterIdx] = bodies[particleId + (w * i)];
			__syncthreads();
			acc = calculateBodyAcceleration(body, currentBodies[filterIdx]);
			atomicAdd(&(accelerations[blockDim.y].x), acc.x);
			atomicAdd(&(accelerations[blockDim.y].y), acc.y);
			atomicAdd(&(accelerations[blockDim.y].z), acc.z);
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			float3 a = acceleration[blockDim.y];
			a.x *= G;
			a.y *= G;
			a.z *= G;
			updateBodyVelocity(a, &dynamic, l_dt);
			updateBodyPosition(velocity, &body, l_dt);
			l_dt += 0.5f;
		}
		__syncthreads();
	}
 //first indx in n particle row, then add together full acceleration and update bodies
}

void beginUniverseSimulation(int numberOfParticles, int partitions, float dt) { //add ability to serialize from past renders.
	size_t allocationSize = sizeof(float4) * numberOfParticles;
	float4 *bodies = malloc(allocationSize);
	float4 *dynamics = malloc(allocationSize);
	float4 *dBodies;
	float4 *dDynamics;
	dim3 blocks(numberOfParticles / partitions, 0, 0);
	dim3 threads(partitions, 0, 0);

	cudaMalloc((void**) &dBodies, allocationSize);
	cudaMalloc((void**) &dDynamics, allocationSize);
	cudaMemcpy(dBodies, bodies, cudaMemcpyHostToDevice);
	cudaMemcpy(dDynamics, dynamics, cudaMemcpyHostToDevice);
	simulateNaive<<<blocks, threads, >>>(d_particles, dt);
}

void resumeUniverseSimulation() {

}
