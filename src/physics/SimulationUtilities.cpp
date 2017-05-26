#include "SimulationUtilities.h"
#include <string>
#include <fstream>
#include <iterator>
#include <algorithm>

static void serializeSimulation(std::string file, float4 *bodies, float4 *dynamics, float3 *accelerations) {
	std::ostream output (file, std::ios::out | std::ios::binary | std::ios::app);
	output.write(reinterpret_cast<const char*> (bodies), sizeof(float4))
}

static void deserializeSimulation(std::string file, float4 *bodies, float4 *dynamics, float3 *accelerations) {

}
