#include "RenderUtil.cuh"

template<typename F>
__global__ void worldToNDC(F* pM, long* N, F* coordinates, F* ndc, long* frustumIndices) {

	long globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
	// if (globalIndex > (*N) - 1) {return;}; // banish extra threads
	F x1, y1, z1, w1;

	F x0 = coordinates[3 * globalIndex + 0];
	F y0 = coordinates[3 * globalIndex + 1];
	F z0 = coordinates[3 * globalIndex + 2];

	x1 = x0 * pM[ 0] + y0 * pM[ 1] + z0 * pM[ 2] + pM[ 3];
	y1 = x0 * pM[ 4] + y0 * pM[ 5] + z0 * pM[ 6] + pM[ 7];
	z1 = x0 * pM[ 8] + y0 * pM[ 9] + z0 * pM[10] + pM[11];
	w1 = x0 * pM[12] + y0 * pM[13] + z0 * pM[14] + pM[15];

	x1 /= w1;
	y1 /= w1;
	z1 /= w1;

	ndc[3 * globalIndex + 0] = x1;
	ndc[3 * globalIndex + 1] = y1;
	ndc[3 * globalIndex + 2] = z1;

	if (x1 >= -1.0 && x1 <= 1.0 &&
		y1 >= -1.0 && y1 <= 1.0 &&
		z1 >= -1.0 && z1 <= 1.0) {frustumIndices[globalIndex] = 1;}
	else {                        frustumIndices[globalIndex] = 0;}
}

template<typename F>
__global__ void clipCoordinates(/*F* coordinates, long* frustumIndices, long* N, F* clippedCoordinates*/) {

	// long globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
	// if (globalIndex > (*N) - 1) {return;}; // banish extra threads
	// printf("%ld\n", frustumIndices[globalIndex]);

	// long cIndex = frustumIndices[globalIndex];
	// clippedCoordinates[3 * globalIndex + 0] = coordinates[3 * cIndex + 0];
	// clippedCoordinates[3 * globalIndex + 1] = coordinates[3 * cIndex + 1];
	// clippedCoordinates[3 * globalIndex + 2] = coordinates[3 * cIndex + 2];
}

struct isLessThanOne
{
	__host__ __device__
	bool operator()(const long x)
	{
		return x < 1;
	}
};

template<typename F>
void render(Camera<F> camera, F* coordinates, long nParticles) {

	// Copy Particle coordinates to the device
	F* d_coordinates;
	size_t size = 3 * nParticles * sizeof(F);
	cudaMalloc((void**) &d_coordinates, size);
	cudaMemcpy(d_coordinates, coordinates, size, cudaMemcpyHostToDevice);

	// Allocate space for NDC coordinates
	F* d_NdcCoordinates;
	size = 3 * nParticles * sizeof(F);
	cudaMalloc((void**) &d_NdcCoordinates, size);

	// Copy particle count to the device
	long* d_NParticles;
	size = sizeof(long);
	cudaMalloc((void**) &d_NParticles, size);
	cudaMemcpy(d_NParticles, &nParticles, size, cudaMemcpyHostToDevice);

	// Copy perspective transformation to device
	Matrix<F> m = camera.getPerspectiveMatrix();
	F perspectiveMatrix[m.nCols * m.nRows];
	m.toArray(perspectiveMatrix);
	size = 16 * sizeof(F);
	F* d_PerspectiveMatrix;
	cudaMalloc((void**) &d_PerspectiveMatrix, size);
	cudaMemcpy(d_PerspectiveMatrix, &perspectiveMatrix, size, cudaMemcpyHostToDevice);

	long nBlocks = nParticles / MAX_THREAD_SIZE;
	nBlocks += (nParticles % MAX_THREAD_SIZE != 0) ? 1 : 0;

	dim3 blockDims(MAX_THREAD_SIZE, 1, 1);
	dim3 gridDims(nBlocks, 1, 1);

	long* d_frustumStencil;
	size = nParticles * sizeof(long);
	cudaMalloc((void**) &d_frustumStencil, size);

	worldToNDC<F><<<gridDims, blockDims>>>(d_PerspectiveMatrix, d_NParticles, d_coordinates, d_NdcCoordinates, d_frustumStencil);
	cudaDeviceSynchronize();

	// Copy frustum stencil to host
	long* frustumStencil = (long*) malloc(nParticles * sizeof(long));
	size = nParticles * sizeof(long);
	cudaMemcpy(frustumStencil, d_frustumStencil, size, cudaMemcpyDeviceToHost);

	// Create 0 to nParticle indices for stencil
	long* frustumStencilIndices = (long*) malloc(nParticles * sizeof(long));
	for (long i = 0; i < nParticles; i++) {
		frustumStencilIndices[i] = 1;
	}
	thrust::exclusive_scan(thrust::host, frustumStencilIndices, frustumStencilIndices + nParticles, frustumStencilIndices);

	// Get frustum indices if they meet the predicate
	long* nParticlesInFrustum = thrust::remove_if(thrust::host, frustumStencilIndices, frustumStencilIndices + nParticles, frustumStencil, isLessThanOne());

	printf("nFrustumParticles: %ld\n", *nParticlesInFrustum);

	nBlocks = *nParticlesInFrustum / MAX_THREAD_SIZE;
	nBlocks += (*nParticlesInFrustum % MAX_THREAD_SIZE != 0) ? 1 : 0;

	blockDims = dim3(MAX_THREAD_SIZE, 1, 1);
	gridDims = dim3(nBlocks, 1, 1);

	printf("%d %d\n", blockDims.x, gridDims.x);

	long* d_frustumStencilIndices;
	size = *nParticlesInFrustum * sizeof(long);
	cudaMalloc((void**) &d_frustumStencilIndices, size);
	cudaMemcpy(d_frustumStencilIndices, &frustumStencilIndices, size, cudaMemcpyHostToDevice);

	long* d_nParticlesInFrustum;
	cudaMalloc((void**) &d_nParticlesInFrustum, sizeof(long));
	cudaMemcpy(d_nParticlesInFrustum, &nParticlesInFrustum, sizeof(long), cudaMemcpyHostToDevice);

	F* d_clippedCoordinates;
	size = 3 * (*nParticlesInFrustum) * sizeof(F);
	cudaMalloc((void**) &d_clippedCoordinates, size);

	clipCoordinates<F><<<gridDims, blockDims>>>(/*d_NdcCoordinates, d_frustumStencilIndices, d_nParticlesInFrustum, d_clippedCoordinates*/);

	size = 3 * (*nParticlesInFrustum) * sizeof(F);
	F* clippedCoordinates = (F*) malloc(size);
	cudaMemcpy(clippedCoordinates, &d_clippedCoordinates, size, cudaMemcpyDeviceToHost);

	/*for (long i = 0; i < *nParticlesInFrustum; i++) {
		printf("%ld: (%f, %f, %f)\n", i, clippedCoordinates[3 * i + 0], clippedCoordinates[3 * i + 1], clippedCoordinates[3 * i + 2]);
	}*/
}



template void render<double>(Camera<double> camera, double* particleCoordinates, long nParticles);
template void render<float>(Camera<float> camera, float* particleCoordinates, long nParticles);
