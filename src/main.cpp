#include <memory>
#include <iostream>
#include <string>
#include "physics/UniverseSimulation.cuh"
#include <cuda.h>
#include "vector_types.h"

namespace Helix {
	int main() { //add save positions
		Helix::UniverseSimSpec<float> spec;
		spec.epochs = 2;
		spec.particles = 512;
		spec.partitions = 32;
		spec.dt = 0.01;
		spec.epsilon = 0.004;

		float4 *ranges = (float4 *)malloc(sizeof(float4) * 2);
		ranges[0].x = 500;
		ranges[0].y = -500;
		ranges[0].z = 50.0f;
		ranges[0].w = -50.0f;
		ranges[1].x = 1000.0f;
		ranges[1].y = 2.0f;
		Helix::beginSimulation<float, float3, float4>(&spec, ranges);
		return 0;
	}
}
