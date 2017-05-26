#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>


// Takes x, y, and z and converts to theta, phi, r
__device__ void rectilinearToSpherical(double* rectilinearPos, double* sphericalPos) {

	double x = rectilinearPos[0];
	double y = rectilinearPos[1];
	double z = rectilinearPos[2];

	double r = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	double phi = atan(y / x);

	sphericalPos[0] = (phi > 0) ? phi : 2 * M_PI + phi;  // Seam of the sphere is in the direction [1 0 0]
	sphericalPos[1] = acos(z / r);                       // Top of the sphere is 0 radians, bottom is 1
	sphericalPos[2] = r;
}

__device__ void sphericalToImage(int* cameraResolution, double* angularPos, int* imagePos) {

	int column = (angularPos[0] / (2 * M_PI)) * (double) cameraResolution[0]; // Implicitly casts double to int
	int row = (angularPos[1] / (1 * M_PI)) * (double) cameraResolution[1];
	*imagePos = column + row * (cameraResolution[0]);
}

__global__ void cudaRender(double* particleCoordinates) {
	printf("Hello world!\n");
}

void render(double* cameraPosition, int* cameraResolution, double* particleCoordinates, long nParticleCoordinates) {

	/* Print the particle coordinates
	for (long i = 0; i < nParticleCoordinates; i++) {
		printf("Input particle positions:\n\n");
		double x = particleCoordinates[i + 0];
		double y = particleCoordinates[i + 1];
		double z = particleCoordinates[i + 2];
		printf("(%lf, %lf, %lf)\n", x, y, z);
	}
	printf("\n");
	*/

	// Copy Particle coordinates to the device
	double* d_particleCoordinates;
	size_t size = 3 * nParticleCoordinates * sizeof(double);
	cudaMalloc((void**) &d_particleCoordinates, size);
	cudaMemcpy(d_particleCoordinates, particleCoordinates, size, cudaMemcpyHostToDevice);

	dim3 blockDims(1, 1, 1);
	dim3 gridDims(1, 1, 1);

	cudaRender<<<blockDims, gridDims>>>(d_particleCoordinates);
	cudaDeviceSynchronize();

}

