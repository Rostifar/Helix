/*
 * UniverseSimulation.h
 *
 *  Created on: May 30, 2017
 *      Author: rostifar
 */
#include <string>
#include "UniverseSimulationKernel.cuh"
#include "ParticleGenerator.cuh"
#include "../hTypes.h"

#ifndef UNIVERSESIMULATION_H_
#define UNIVERSESIMULATION_H_

namespace Helix {

template<typename F>
class UniSimFmt {
public:
	Vector2<F> rLim;
	Vector2<F> vLim;
	Vector2<F> aLim;
	Vector2<F> mLim;
	Vector4<F> generationType;
	const auto len = 12;

	UniSimFmt() {}
	UniSimFmt(Vector2<F> _rLim, Vector2<F> _vLim, Vector2<F> _aLim, Vector2<F> _mLim, Vector4<F> _generationType)
		: rLim(_rLim), vLim(_vLim), aLim(_aLim), mLim(_mLim), generationType(_generationType) {}

	virtual ~UniSimFmt(){}

	F *toCudaFmt() {
		F *arr = new F[len];
		rLim.map(arr, 0);
		vLim.map(arr, 2);
		aLim.map(arr, 4);
		mLim.map(arr, 6);
		generationType.map(arr, 8);
		return arr;
	}

	void toCudaFmt(F *arr, int i) {
		rLim.map(arr, i);
		vLim.map(arr, i + 2);
		aLim.map(arr, i + 4);
		mLim.map(arr, i + 6);
		generationType.map(arr, i + 8);
	}

	void toHostFmt(F *arr, int i) {
		for (int q = 0; q < 2; q++, i++) {
			rLim[q] = arr[i];
			vLim[q] = arr[i + 2];
			aLim[q] = arr[i + 4];
			mLim[q] = arr[i + 6];
			generationType[q] = arr[i + 8];
			generationType[q] = arr[i + 10];
		}
	}
};

template<typename F>
class UniParticle {
public:
	Vector3<F>  r;
	Vector3<F>  velocity;
	Vector3<F>  acceleration;
	F	   		mass;
	const static auto len  = 10;

	UniParticle(Vector3<F> _r, Vector3<F> _v, Vector3<F> _acc, F _mass)
		: r(_r), velocity(_v), acceleration(_acc), mass(_mass) {}

	virtual ~UniParticle(){}

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
			r[q] 			= arr[i];
			velocity[q]		= arr[i + 3];
			acceleration[q] = arr[i + 6];
		}
	}
};

template<typename F>
class UniverseSimulation {
public:
	UniverseSimulation(int _nParticles, int _epsilon, F _dt);
	void addGenerationLimits(UniSimFmt *limits);
	void beginUniverseSimulation();
	void resumeUniverseSimulation(std::string file);
	void onEpochCompletion(UniParticle<F> *particles);
	virtual ~UniverseSimulation();

private:
	int		n;
	int		epsilon;
	F		dt;
	auto	epochs 		= 0;
	bool	limitsSet	= false;

	UniLimitFmt<F> limits;
	void pregenerateLimits();
	
	//Rough estimates for the amount of celestial bodies in the observable universe.
	const F universeRadius		= 8.8E23; //aka. 28.5 gpc
	const F stars				= 10E21;
	const F planets				= 1E24;
	const F galaxies			= 10E9;
	const F celestrialBodies	= stars + planets;
	const F fastestBodies		= 700; // km/s
};

template<typename F>
UniverseSimulation<F>::UniverseSimulation(int _nParticles, int _epsilon, F _dt, Platform platform = Platform::CPU) {
	n	= nParticles;
	epsilon	= _epsilon;
	dt	= _dt;
}

template<typename F>
void UniverseSimulation<F>::addGenerationLimits(UniLimitFmt<F> *_limits) {
	limits = _limits;
	limitsSet = true;
}

template<typename F>
void UniverseSimulation<F>::pregenerateLimits() {
	F scalar	= celestrialBodies / universeRadius;
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

template<typename F>
void UniverseSimulation<F>::beginUniverseSimulation() {
	if (!limitsSet) pregenerateLimits();
	if (!gpuSupported()) return;//implement me!
	F *particles, *dParticles;
	densityParticleGeneration<F>(&limits, n, particles, dParticles);
}

template<typename F>
void UniverseSimulation<F>::onEpochCompletion(UniParticle<F> *particles) {

}

template<typename F>
void UniverseSimulation<F>::resumeUniverseSimulation(std::string file) {

}

}

#endif /* UNIVERSESIMULATION_H_ */
