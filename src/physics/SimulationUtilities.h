/*
 * SimulationUtilities.h
 *
 *  Created on: May 23, 2017
 *      Author: rostifar
 */

#ifndef SIMULATIONUTILITIES_H_
#define SIMULATIONUTILITIES_H_
#include <string>
#include <memory>

typedef struct UniverseSimSpec {
	int nParticles;
	int partitions;
	int epochs;
	float4 *ranges;
};
#endif /* SIMULATIONUTILITIES_H_ */
