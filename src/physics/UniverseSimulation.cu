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

template<typename F, typename F3, typename F4>
__device__ F3 updateBodyVelocity(F3 a, F4 v, F dt) { //velocity verlet
	F3 newV;
	newV.x = v.x + 0.5 * a.x * dt;
	newV.y = v.y + 0.5 * a.y * dt;
	newV.z = v.z + 0.5 * a.z * dt;
	return newV;
}

template<typename F>
__global__ void simulateNaive(F4 *bodies, F4 *dynamics, F3 *accelerations, F _dt, F _epsilon, int n_particles) {
	F dt = _dt;
	int particleId = blockDim.x * blockIdx.x + threadIdx.x;
	F4 body = bodies[particleId];
	F4 dynamic = dynamics[particleId];
	F3 velocity = {dynamic.x, dynamic.y, dynamic.z};
	F3 acceleration = accelerations[particleId];
	__shared__ F4 *interactingBodies;

	F3 vHalf = updateBodyVelocity(acceleration, velocity, dt);
	body.x += vHalf.x * dt;
	body.y += vHalf.y * dt;
	body.z += vHalf.z * dt;
	for (int i = 0; i < n_particles; i += blockDim.x) {
		interactingBodies[threadIdx.x] = bodies[threadIdx.x + i];
		__syncthreads();
		calculatePartitionAcceleration<F, F3, F4>(body, &acceleration, interactingBodies, _epsilon);
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

template<typename F>
void beginSimulation(UniverseSimSpec<F> *_specs, Helix::FN<F> *_ranges) {
	F 	 *dBodies;
	dim3 blocks 				(_specs->particles / _specs->partitions, 0, 0);
	dim3 threads				(_specs->partitions, 0, 0);
	Helix::FN<F> bodies			(_specs->particles * _specs->offset);
	Helix::generateParticles<F> (_ranges->vec, bodies.vec, dBodies, blocks, threads, _specs->offset);

	for (int i = 0; i < spec->epochs; i++) {
		simulateNaive<F, F3, F4><<<blocks, threads, sizeof(F4) * spec->partitions>>>(dBodies, dDynamics, dAccelerations, dt, epsilon, spec->particles);
		cudaMemcpy(bodies, dBodies, allocationSize,cudaMemcpyDeviceToHost); //copy back to save to binary file
		cudaMemcpy(dynamics, dDynamics, allocationSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(accelerations, dAccelerations, allocationSize, cudaMemcpyDeviceToHost); //yo lance sucks
	}
}
template void beginSimulation <float, float3, float4>(UniverseSimSpec<float> *, float4 *);
template void beginSimulation <double, double3, double4>(UniverseSimSpec<double> *, double4 *);
}
