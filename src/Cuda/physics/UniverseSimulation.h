/*
 * UniverseSimulation.h
 *
 *  Created on: May 30, 2017
 *      Author: rostifar
 */
#include <string>
#include "UniverseSimulationKernel.cuh"
#include "../hTypes.h"

#ifndef UNIVERSESIMULATION_H_
#define UNIVERSESIMULATION_H_

namespace Helix {

//Holds standard parameters for every simulation, either new or resumed
template<typename F>
struct UniSimParams {
	F epsilon, dt, partitions, nParticles;
};

template<typename F>
class UniSimFmt {
public:
	Vector2<F> rLim, vLim, aLim, mLim;
	Vector4<F> genType;
	bool limitsSet = false;
	const auto rawSize = sizeof(Vector2<F>) * 4 + sizeof(Vector4<F>);
	const static auto len = 12;

	UniSimFmt() {}
	virtual ~UniSimFmt() {}

	UniSimFmt(Vector2<F> _rLim, Vector2<F> _vLim, Vector2<F> _aLim, Vector2<F> _mLim, Vector4<F> _genType) {
		rLim = _rLim;a
		vLim = _vLim;
		aLim = _aLim;
		mLim = _mLim;
		genType = _genType;
		limitsSet = true;
	}

	F *toCudaFmt() {
		F *arr = new F[len];
		rLim.map(arr, 0);
		vLim.map(arr, 2);
		aLim.map(arr, 4);
		mLim.map(arr, 6);
		genType.map(arr, 8);
		return arr;
	}

	void toCudaFmt(F *arr, int i) {
		rLim.map(arr, i);
		vLim.map(arr, i + 2);
		aLim.map(arr, i + 4);
		mLim.map(arr, i + 6);
		genType.map(arr, i + 8);
	}

	void toHostFmt(F *arr, int i) {
		for (int q = 0; q < 2; q++, i++) {
			rLim[q] = arr[i];
			vLim[q] = arr[i + 2];
			aLim[q] = arr[i + 4];
			mLim[q] = arr[i + 6];
			genType[q] = arr[i + 8];
			genType[q] = arr[i + 10];
		}
	}
};

template<typename F>
class UniParticle {
public:
	Vector3<F> r, velocity, acceleration;
	F mass;
	const static auto particleSize = sizeof(Vector3<F> ) * 3 + sizeof(F);
	const static auto len = 10;
	const static auto commonDiff = 3;

	UniParticle() {}
	
	virtual ~UniParticle() {}

	UniParticle(Vector3<F> _r, Vector3<F> _v, Vector3<F> _acc, F _mass) :
			r(_r), velocity(_v), acceleration(_acc), mass(_mass) {}

	F *toCudaFmt() {
		F *arr = new F[len];
		r.map(arr, 0);
		velocity.map(arr, 3);
		acceleration.map(arr, 6);
		arr[9] = mass;
		return arr;
	}

	void toCudaFmt(F *arr, int i) {
		r.map(arr, i);
		velocity.map(arr, i + 3);
		acceleration.map(arr, i + 6);
		arr[i + 9] = mass;
	}

	void toHostFmt(F *arr, int i) {
		mass = arr[i + 9];
		for (int q = 0; q < 3; q++, i++) {
			r[q] = arr[i];
			velocity[q] = arr[i + 3];
			acceleration[q] = arr[i + 6];
		}
	}

	static void massCudaFmt(F *arr, UniParticle<F> *particles, const int n) {
		for (int i = 0; i < n; i++) {
			particles[i].toCudaFmt(arr, i);
		}
	}

	static void massHostFmt(F *arr, UniParticle<F> *particles, const int n) {
		for (int i = 0, q = 0; i < n; i++, q += len) {
			particles[i].toHostFmt(arr, q);
		}
	}
};

template<typename F>
class UniverseSimulation {
public:
	enum class Algorithm {
		NAIVE
	};

	UniverseSimulation() {
		epsilon = defEpsilon;
		epochs = defEpochs;
		n = defParticles;
		dt = defDt;
		partitions = defPartitions;
		algorithm = Algorithm::NAIVE;
		particles = UniParticle<F>[n];
	}

	UniverseSimulation(UniSimParams<F> *params, const int _epochs, Algorithm alg = Algorithm::NAIVE) {
		epochs = _epochs;
		algorithm = alg;
		particles = UniParticle<F>[n];
	}

	void addSimulationLimits(UniSimFmt<F> *_limits) {
		limits.rLim = _limits->rLim;
		limits.vLim = _limits->vLim;
		limits.aLim = _limits->aLim;
		limits.mLim = _limits->mLim;
		limits.genType = _limits->genType;
	}
	void beginUniverseSimulation() {
		checkSimulationParams();
		
		F *dParticles;
		KernelDimensions dims;
		dims.blocks = n / partitions;
		dims.threads = partitions;

		generateDistributedParticles<F>(&limits, particles, dParticles, &dims, n);
	}
	void resumeUniverseSimulation(std::string file);
	void onEpochCompletion(UniParticle<F> *particles);
	virtual ~UniverseSimulation();

private:
//parameter minimums and default
	const auto minParticles 	= 64;
	const auto minEpochs		= 1;
	const auto defParticles 	= 1024;
	const auto defEpochs		= 3;
	const auto defPartitions	= 32;
	const F	minDt				= 0.0001;
	const F minEpsilon			= 0.0001;
	const F defDt				= 0.01;
	const F defEpsilon 			= 0.04;

//Rough estimates for the amount of celestial bodies in the observable universe.
	const F universeRadius		= 8.8E23; //aka. 28.5 gpc
	const F stars				= 10E21;
	const F planets				= 1E24;
	const F galaxies			= 10E9;
	const F fastestBodies		= 700; // km/s

	F celestrialBodies = stars + planets;

	UniSimParams<F> params;
	int epochs;
	int simId = 12; //TODO: implement simulation id assignment.
	Algorithm algorithm;

	UniSimFmt<F> limits;
	UniParticle<F> *particles;

void pregenerateLimits() {
	F scalar = celestrialBodies / universeRadius;
	F estimateR = n / scalar;
	limits.rLim.x = estimateR;
	limits.rLim.y = -estimateR;

	limits.vLim.x = 120;
	limits.vLim.y = 120;
	limits.aLim.x = 0;
	limits.aLim.y = 0;

	limits.mLim.x = 12 * 1.99E30;
	limits.mLim.y = 1E3;

	limits.generationType.x = UNIFORM;
	limits.generationType.y = GAUSSIAN;
	limits.generaitonType.z = NONE;
	limits.generationType.w = GUASSIAN;
}

void checkSimulationParams() {
	if (!limits.limitsSet) {
		printf("%s %i %s \n", "WARNING. No limits were defined in simulation: ",
				simId, ". Configuring simulation using min values.");
		pregenerateLimits();
	}

	if (n < minParticles) {
		printf("%s, %i, %s, \n",
				"WARNING. Selected particle count is below minimum value in simulation: ",
				simId, ". Configuring using min values.");
		n = minParticles;
	}

	if (epochs < minEpochs) {
		printf("%s, %i, %s, \n",
				"WARNING. Selected epoch count is below minimum value in simulation: ",
				simId, ". Configuring using min values.");
		epochs = minEpochs;
	}

	if (dt < minDt) {
		printf("%s, %i, %s, \n",
				"WARNING. Selected dt value is below minimum value in simulation: ",
				simId, ". Configuring using min values.");
		dt = minDt;
	}
	if (dt < minEpsilon) {
		printf("%s, %i, %s, \n",
				"WARNING. Selected epsilon value is below minimum value in simulation: ",
				simId, ". Configuring using min values.");
		dt = minEpsilon;
	}
}

void generateParticles() {}
};

}
#endif /* UNIVERSESIMULATION_H_ */
