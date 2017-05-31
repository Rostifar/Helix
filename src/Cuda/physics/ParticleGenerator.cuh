/*
 * ParticleGenerator.cuh
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */
#include "../Types.cuh"

#ifndef PARTICLEGENERATOR_CUH_
#define PARTICLEGENERATOR_CUH_

namespace Helix {

enum Platform {
	CPU,
	GPU
}


template<typename F>
void distributionGeneration(F4<F> *_particles, F4<F> *_dParticles, GenerationLimits<F4<F>> *limits, int nParticles, dim3 *blocks, dim3 *threads, int offset, bool localCpy = false);
}



#endif /* PARTICLEGENERATOR_CUH_ */
