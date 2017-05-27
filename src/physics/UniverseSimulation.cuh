/*
 * UniverseSimulation.cuh
 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */

#ifndef UNIVERSESIMULATION_CUH_
#define UNIVERSESIMULATION_CUH_

typedef struct UniverseSimSpec {
	int particles;
	int partitions;
	int epochs;
};

template<class F, class F3, class F4>
void beginSimulation(UniverseSimSpec *specs, F _dt, F _epsilon);

template<class F, class F3, class F4>
void resumeSimulation();

#endif /* UNIVERSESIMULATION_CUH_ */
