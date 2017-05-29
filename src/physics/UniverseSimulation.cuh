/*
 * UniverseSimulation.cuh
 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */
#include "UniverseParticleGenerator.cuh"
#include "../DataTypes.h"

#ifndef UNIVERSESIMULATION_CUH_
#define UNIVERSESIMULATION_CUH_

namespace Helix {

template <typename F>
struct UniverseSimulator {
	const int offset = 10;
	int particles, partitions, epochs;
	F dt, epsilon;
	DataFmt fmt;
	F *limits;
};

void beginSimulation(Helix<F> *_ranges, );

}

#endif /* UNIVERSESIMULATION_CUH_ */
