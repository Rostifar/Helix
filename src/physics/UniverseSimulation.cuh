/*
 * UniverseSimulation.cuh
 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */

#ifndef UNIVERSESIMULATION_CUH_
#define UNIVERSESIMULATION_CUH_

namespace Helix {


template <typename F, typename F4>
struct UniverseSimSpec {
	int particles;
	int partitions;
	int epochs;
	F dt;
	F epsilon;
	F4 range;
};

template<typename F, class F3, class F4>
void beginSimulation(UniverseSimSpec<F> *specs, F4 *_ranges);
}

#endif /* UNIVERSESIMULATION_CUH_ */
