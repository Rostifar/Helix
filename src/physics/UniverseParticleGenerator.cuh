/*
 * UniverseParticleGenerator.cuh
 *
 *  Created on: May 28, 2017
 *      Author: rostifar
 */

#include "UniverseSimulation.cuh"
#include "../CudaUtilities.cuh"

#ifndef UNIVERSEPARTICLEGENERATOR_CUH_
#define UNIVERSEPARTICLEGENERATOR_CUH_

namespace Helix {

template<typename F>
struct UniverseGeneratorSpec {
	F *limits;
	int typeOffset;
	int stride;
};

template<typename F>
void generateParticles(F *bodies, F *dBodies, UniverseGeneratorSpec *uniSpec, dim3 blocks, dim3 threads);

};

#endif /* UNIVERSEPARTICLEGENERATOR_CUH_ */
