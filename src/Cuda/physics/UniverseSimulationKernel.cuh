/*
 * UniverseSimulation.cuh
 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */
#include "UniverseSimulation.h"
#include "../CudaUtilities.cuh"

#ifndef UNIVERSESIMULATION_CUH_
#define UNIVERSESIMULATION_CUH_

/*
 *	N-BODY UNIVERSE SIMULATION [Newtonian]
 *	--------------------------------------
 *	* Observable universe has a diameter of 8.8 x 10^23 km. Conversely, we will determine a universe to be
 *		n / 2^23
 *	*
 * */

namespace Helix {

template<typename F>
void generateDistributedParticles( UniSimFmt<F> *_limits, UniParticle<F> *_particles, F *_dParticles, KernelDimensions *dims, int n, bool cpyLocal = false );

template<typename F>
void startUniverseKernel(F epsilon, F dt, int n, int p, int epochs, UniLimitFmt<F> limits);

}

#endif /* UNIVERSESIMULATION_CUH_ */
