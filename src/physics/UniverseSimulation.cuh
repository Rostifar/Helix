/*
 * UniverseSimulation.cuh
 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */

#ifndef UNIVERSESIMULATION_CUH_
#define UNIVERSESIMULATION_CUH_

template <typename F>
struct UniverseSimSpec {
	int particles;
	int partitions;
	int epochs;
	F dt;
	F epsilon;
};

template<typename F, class F3, class F4>
void beginSimulation(UniverseSimSpec<F> *specs, F _dt, F _epsilon, F4 *_ranges);

#endif /* UNIVERSESIMULATION_CUH_ */
