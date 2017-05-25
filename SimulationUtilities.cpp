#include "SimulationUtilities.h"

void handleError(SIMULATION_ERROR e) {
	char msg[];
	if (e == IMPROPER_PARTICLE_COUNT) {
		msg = "error. the inputed number of particles and your given partition does not follow 2^n.";
	}
	else if (e == MAX_THREAD_COUNT) {
		msg = "error. the selected block/thread config is too large. your dimensions will be reconfigured temporarily.";
	}
	printf("%s", msg);
}
