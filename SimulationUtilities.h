/*
 * SimulationUtilities.h
 *
 *  Created on: May 23, 2017
 *      Author: rostifar
 */

#ifndef SIMULATIONUTILITIES_H_
#define SIMULATIONUTILITIES_H_

typedef enum SIMULATION_ERROR {
	NOT_PEFECT_SQUARE,
	MAX_THREAD_COUNT,
	SUCCESS
};

void handleErrors(SIMULATION_ERROR e);





#endif /* SIMULATIONUTILITIES_H_ */
