#include <memory>
#include <iostream>
#include <string>
#include "physics/UniverseSimulation.cuh"
#include <cuda.h>
#include "vector_types.h"
#include "DataTypes.h"

namespace Helix {
	int main() { //add save positions
		FN<float> ranges(9);
		ranges =
			{
				2.3E9, -2.3E9, UNIFORM,
				1.0E3, -1.0E3, GAUSSIAN,
				1.0E3, 1.0E-2, GAUSSIAN
			};

		UniverseSimulator<float> simulator(1024, 32, DataFmt::UNIFORM, 2, 0.004, 0.004);

		return 0;
	}
}
