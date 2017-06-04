#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "src/render/RenderUtil.cuh"

int main(int argc, char **argv) {

	// Inputs ------------------------------------------------------------------
	long nParticles = 100;
	float particleRadius = 0.001;
	int kernelSize = 11;

	Camera<float> camera;
	camera.position    = Vector<float>(0, 0, 0);
	camera.gaze        = Vector<float>(1, 0, 0);
	camera.up          = Vector<float>(0, 1, 0);
	camera.xFov        = 90;
	camera.yFov        = 90;
	camera.nCols       = 2000;
	camera.nRows       = 2000;
	camera.maxClipping = 300.0;
	camera.setMinClipping(kernelSize, particleRadius);
	// Inputs -----------------------------------------------------------------

	// Fill random coordinates for particles
	int maxDim = 100;
	srand(time(NULL));
	float coordinates[3 * nParticles];
	for (long i = 0; i < 3 * nParticles; i++) {
		coordinates[i] = (float) ((rand() % (2 * maxDim + 1)) - maxDim);
	}

	render<float>(camera, coordinates, nParticles);

	return 0;
}

