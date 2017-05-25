/*
 * SimulationUtilities.h
 *
 *  Created on: May 23, 2017
 *      Author: rostifar
 */

#ifndef SIMULATIONUTILITIES_H_
#define SIMULATIONUTILITIES_H_
#include <string>

typedef enum SIMULATION_ERROR {
	NOT_PEFECT_SQUARE,
	MAX_THREAD_COUNT,
	SUCCESS
};

void handleErrors(SIMULATION_ERROR e);
void static serializeSimulation(std::string file, float4 *bodies, float4 *dynamics, float3 *accelerations);
void static deserializeSimulation(std::string file,  float4 *bodies, float4 *dynamics, float3 *accelerations);

#endif /* SIMULATIONUTILITIES_H_ */
