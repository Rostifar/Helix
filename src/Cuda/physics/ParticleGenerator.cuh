/*
 * ParticleGenerator.cuh
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */
#include "../Types.cuh"
#include "UniverseSimulation.h"

#ifndef PARTICLEGENERATOR_CUH_
#define PARTICLEGENERATOR_CUH_

namespace Helix {

typedef enum Platform {
	CPU,
	GPU
};

template<typename F>
void densityParticleGeneration(UniLimitFmt<F> *limits, int n, F *_particles, F *_dParticles);

template<typename F>
void distributionGeneration(F4<F> *_particles, F4<F> *_dParticles, GenerationLimits<F4<F>> *limits, int nParticles, dim3 *blocks, dim3 *threads, int offset, bool localCpy = false);
}



#endif /* PARTICLEGENERATOR_CUH_ */
