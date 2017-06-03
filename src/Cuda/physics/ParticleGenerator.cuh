/*
 * ParticleGenerator.cuh
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */
#include "UniverseSimulation.h"

#ifndef PARTICLEGENERATOR_CUH_
#define PARTICLEGENERATOR_CUH_

namespace Helix {

template<typename F>
void densityParticleGeneration(UniSimFmt<F> *limits, int n, F *_particles, F *_dParticles);

#endif /* PARTICLEGENERATOR_CUH_ */
