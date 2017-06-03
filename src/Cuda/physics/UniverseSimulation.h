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
	F	epsilon;
	F	dt;
	F	partitions;
	F	nParticles;
};

template<typename F>
class UniSimFmt {
public:
	Vector2<F>	rLim;
	Vector2<F>	vLim;
	Vector2<F>	aLim;
	Vector2<F>	mLim;


	Vector4<F> generationType;
	bool limitsSet = false;
	const auto rawSize = sizeof(Vector2<F> ) * 4 + sizeof(Vector4<F> );
	const auto len = 12;
	const auto commonDiff = 2;
	const F *header = { 5, 2, 2, 2, 2, 4 };

	UniSimFmt() {
	}
	virtual ~UniSimFmt() {
	}

	UniSimFmt(Vector2<F> _rLim, Vector2<F> _vLim, Vector2<F> _aLim,
			Vector2<F> _mLim, Vector4<F> _generationType) {
		rLim = _rLim;
		vLim = _vLim;
		aLim = _aLim;
		mLim = _mLim;
		generationType = _generationType;
		limitsSet = true;
	}

	F *toCudaFmt() {
		F *arr = new F[len];
		rLim.map(arr, 0);
		vLim.map(arr, commonDiff);
		aLim.map(arr, commonDiff * 2);
		mLim.map(arr, commonDiff * 3);
		generationType.map(arr, commonDiff * 4);
		return arr;
	}

	void toCudaFmt(F *arr, int i) {
		rLim.map(arr, i);
		vLim.map(arr, i + commonDiff);
		aLim.map(arr, i + (commonDiff * 2));
		mLim.map(arr, i + (commonDiff * 3));
		generationType.map(arr, i + (commonDiff * 4));
	}

	void toHostFmt(F *arr, int i) {
		for (int q = 0; q < commonDiff; q++, i++) {
			rLim[q] = arr[i];
			vLim[q] = arr[i + commonDiff];
			aLim[q] = arr[i + (commonDiff * 2)];
			mLim[q] = arr[i + (commonDiff * 3)];
			generationType[q] = arr[i + (commonDiff * 4)];
			generationType[q] = arr[i + (commonDiff * 5)];
		}
	}
};

template<typename F>
class UniParticle {
public:
	Vector3<F> r;
	Vector3<F> velocity;
	Vector3<F> acceleration;
	F mass;

	const static auto RAW_PARTICLE_SIZE = sizeof(Vector3<F> ) * 3 + sizeof(F);
	const static auto len = 10;
	const static auto commonDiff = 3;

	UniParticle() {
	}

	virtual ~UniParticle() {
	}

	UniParticle(Vector3<F> _r, Vector3<F> _v, Vector3<F> _acc, F _mass) :
			r(_r), velocity(_v), acceleration(_acc), mass(_mass) {
	}

	F *toCudaFmt() {
		F *arr = new F[len];
		r.map(arr, 0);
		velocity.map(arr, commonDiff);
		acceleration.map(arr, commonDiff * 2);
		arr[commonDiff * 3] = mass;
		return arr;
	}

	void toCudaFmt(F *arr, int i) {
		r.map(arr, i);
		velocity.map(arr, i + commonDiff);
		acceleration.map(arr, i + (commonDiff * 2));
		arr[i + (commonDiff * 3)] = mass;
	}

	void toHostFmt(F *arr, int i) {
		mass = arr[i + (commonDiff * 3)];
		for (int q = 0; q < commonDiff; q++, i++) {
			r[q] = arr[i];
			velocity[q] = arr[i + commonDiff];
			acceleration[q] = arr[i + (commonDiff * 2)];
		}
	}

	static void massCudaFmt(F *arr, UniParticle *particles, const int n) {
		for (int i = 0; i < n; i++) {
			particles[i].toCudaFmt(arr, i);
		}
	}
};

template<typename F>
class UniverseSimulation {
public:
	enum class Algorithm {
		NAIVE
	};

	UniverseSimulation(UniSimParams<F> *params,
			Algorithm alg = Algorithm::NAIVE) {
		epsilon = params->epsilon;
		n = params->nParticles;
		partitions = params->partitions;
		dt = params->dt;
	}

	void addSimulationLimits(UniSimFmt<F> *_limits) {
		limits.rLim = _limits->rLim;
		limits.vLim = _limits->vLim;
		limits.aLim = _limits->aLim;
		limits.mLim = _limits->mLim;
		limits.generationType = _limits->generationType;
	}
	void beginUniverseSimulation() {
		checkSimulationParams();
		if (!gpuSupported())
			return; //implement me!

		F *cudaParticles = new F[n * UniParticle<F>::len];
		F *dParticles = new F[n * UniParticle<F>::len];
		KernelDimensions dims;
	dims.blocks = n /

	generateDistributedParticles<F>( &limits, particles, dParticles, )
}
void resumeUniverseSimulation(std::string file);
void onEpochCompletion(UniParticle<F> *particles);
virtual ~UniverseSimulation();

private:
//parameter minimums and default
auto const minParticles = 64;
auto const minEpochs = 1;
F const minDt = 0.0001;
F const minEpsilon = 0.0001;
auto const defParticles = 1024;
auto const defEpochs = 3;
F const defDt = 0.01;
F const defEpsilon = 0.04;

//Rough estimates for the amount of celestial bodies in the observable universe.
F const universeRadius = 8.8E23; //aka. 28.5 gpc
F const stars = 10E21;
F const planets = 1E24;
F const galaxies = 10E9;
F const fastestBodies = 700; // km/s

F celestrialBodies = stars + planets;
int epochs;
int simId = 12; //TODO: implement simulation id assignment.
int n;
int epsilon;
int partitions;
F dt;
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
	limits.generationType.w = POISSON;
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

void generateParticles() {

}
};

}
#endif /* UNIVERSESIMULATION_H_ */
