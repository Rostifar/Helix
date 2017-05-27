#include "SimulationUtilities.h"
#include <memory>
#include <iostream>
#include <string>

int main() { //add save positions
	int option;
	std::cout << "WELCOME TO THE GREATEST APP IN THE WORLD." << std::endl;
	std::cout << "Options: " << std::endl;
	std::cout << "[1] Start a new simulation." << std::endl;
	std::cout << "[2] Resume simulation" << std::endl;
	std::cout << "....." << std::endl;
	std::cin >> option;

	if (option == 0) {
		UniverseSimSpec *spec = new UniverseSimSpec();
		std::cout << "How many particles:" << std::endl;
		std::cin >> spec->epochs;
		std::cout << "How many partitions:" << std::endl;
		std::cin >> spec->partitions;
		std::cout << "How many epochs:" << std::endl;
		std::cin >> spec->epochs;
		std::cout << "What ranges: [position_high, position_low, velocity_high, velocity_low, mass_high, mass_low]" << std::endl;


	}
	else if (option == 0) {
		return 0;
	}
	else {
		return 0;
	}
	cudaDeviceSynchronize();
	return 0;
}
