#include <memory>
#include <iostream>
#include <string>
#include "Cuda/physics/UniverseSimulation.h"
#include <cuda.h>
#include "vector_types.h"
#include "Types.cuh"
#include "Cuda/CudaUtilities.cuh"

int main() {
	Helix::UniverseSimulation<float>(1024, 0.02, 0.01);



	Helix::F4<float> ceilings   = Helix::makeF4(2.3E9, 1.0E3, 0, 1.0E3);
	Helix::F4<float> floors     = Helix::makeF4(-2.3E9, -1.0E3, 0, 1.0E-2);
	Helix::F4<int>   distrTypes = Helix::makeF4(UNIFORM, GAUSSIAN, NONE, GAUSSIAN);
	int	             *strides   = {3, 1, 3, 3};

	Helix::GenerationLimits<Helix::F4<float>> limits(12, ceilings, floors, distrTypes, strides);
	Helix::UniverseParams<float> params(0.5, 0.04, 1024, 32, 2, 3);
	Helix::beginUniSimNaive<float>(&params, &limits);
	return 0;
}

