
#include <stdio.h>
#include <math.h>

void projectToCamera(double* cameraPositon, int* cameraResolution, double* particlePosition, int* imagePosition);
void rectilinearToSpherical(double*, double*);
void sphericalToImage(int* cameraResolution, double* angularPos, int* imagePos);
void render(double* cameraPosition, int* cameraResolution, double* particleCoordinates, long nParticleCoordinates);

int main(int argc, char **argv){

	int cameraResolution[2] = {4, 2}; // Cols by Rows

	long nParticleCoordinates = 1;
	double particleCoordinates[3] = {1, 0, 0};
	double cameraPosition[3] = {0, 0, 0};

	render(cameraPosition, cameraResolution, particleCoordinates, nParticleCoordinates);

	return 0;
}
