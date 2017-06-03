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
	bool limitsSet = false;
	const auto RAW_FMT_SIZE = sizeof(Vector2<F>) * 4 + sizeof(Vector4<F>);
	const auto len			= 12;

	UniSimFmt() {}

	UniSimFmt(Vector2<F> _rLim, Vector2<F> _vLim, Vector2<F> _aLim, Vector2<F> _mLim, Vector4<F> _generationType)
		: rLim(_rLim), vLim(_vLim), aLim(_aLim), mLim(_mLim), generationType(_generationType) {
		limitsSet = true;
	}

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

	const static auto RAW_PARTICLE_SIZE = sizeof(Vector3<F>) * 3 + sizeof(F);
	const static auto len  				= 10;
	const static auto commonDifference	= 3;

	UniParticle() {}

	UniParticle(Vector3<F> _r, Vector3<F> _v, Vector3<F> _acc, F _mass)
		: r(_r), velocity(_v), acceleration(_acc), mass(_mass) {}

	virtual ~UniParticle(){}

	F *toCudaFmt() {
		F *arr = new F[len];
		r.map(arr, 0);
		velocity.map(arr, commonDifference);
		acceleration.map(arr, commonDifference * 2);
		arr[commonDifference * 3] = mass;
		return arr;
	}

	void toCudaFmt(F *arr, int i) {
		r.map(arr, i);
		velocity.map(arr, i + commonDifference);
		acceleration.map(arr, i + (commonDifference * 2));
		arr[i + (commonDifference * 3)] = mass;
	}

	void toHostFmt(F *arr, int i) {
		mass = arr[i + (commonDifference * 3)];
		for (int q = 0; q < commonDifference; q++, i++) {
			r[q] 			= arr[i];
			velocity[q]		= arr[i + commonDifference];
			acceleration[q] = arr[i + (commonDifference * 2)];
		}
	}

	static void massCudaMap(F *arr, UniParticle *particles) {
		assert(sizeof(arr) == sizeof(particles))
	}
};

template<typename F>
class UniverseSimulation {
public:
	enum class Algorithm {
		NAIVE
	};

	UniverseSimulation(int _nParticles = DEF_PARTICLES, int _epsilon = DEF_EPSILON, int _epochs = DEF_EPOCHS, F _dt = DEF_DT, Algorithm alg = Algorithm::NAIVE) {
		n			= _nParticles;
		epsilon		= _epsilon;
		dt			= _dt;
		epochs		= _epochs;
		particles	= new UniParticle<F>[n];
		algorithm	= alg;
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
		if (!gpuSupported()) return; //implement me!

		F *dParticles;
		densityParticleGeneration<F>(&limits, n, particles, dParticles);
	}
	void resumeUniverseSimulation(std::string file);
	void onEpochCompletion(UniParticle<F> *particles);
	virtual ~UniverseSimulation();

private:
	//parameter minimums and default
	auto const	MIN_PARTICLES	= 64;
	auto const	MIN_EPOCHS		= 1;
	F	 const	MIN_DT			= 0.0001;
	F	 const	MIN_EPSILON		= 0.0001;
	auto const	DEF_PARTICLES	= 1024;
	auto const	DEF_EPOCHS		= 3;
	F	 const 	DEF_DT			= 0.01;
	F	 const	DEF_EPSILON		= 0.04;

	//Rough estimates for the amount of celestial bodies in the observable universe.
	F	 const	universeRadius	= 8.8E23; //aka. 28.5 gpc
	F	 const	stars			= 10E21;
	F	 const	planets			= 1E24;
	F	 const	galaxies		= 10E9;
	F		  	celestrialBodies= stars + planets;
	F	 const	fastestBodies	= 700; // km/s

	auto		epochs;
	int			simId			= 12; //TODO: implement simulation id assignment.
	int			n;
	int			epsilon;
	F			dt;
	Algorithm	algorithm;

	UniSimFmt<F>	limits;
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
			printf("%s %i %s \n", "WARNING. No limits were defined in simulation: ", simId, ". Configuring simulation using min values.");
			pregenerateLimits();
		}

		if (n < MIN_PARTICLES) {
			printf("%s, %i, %s, \n", "WARNING. Selected particle count is below minimum value in simulation: ", simId, ". Configuring using min values.");
			n = MIN_PARTICLES;
		}

		if (epochs < MIN_EPOCHS) {
			printf("%s, %i, %s, \n", "WARNING. Selected epoch count is below minimum value in simulation: ", simId, ". Configuring using min values.");
			epochs = MIN_EPOCHS;
		}

		if (dt < MIN_DT) {
			printf("%s, %i, %s, \n", "WARNING. Selected dt value is below minimum value in simulation: ", simId, ". Configuring using min values.");
			dt = MIN_DT;
		}
		if (dt < MIN_EPSILON) {
			printf("%s, %i, %s, \n", "WARNING. Selected epsilon value is below minimum value in simulation: ", simId, ". Configuring using min values.");
			dt = MIN_DT;
		}
	}
};

}
#endif /* UNIVERSESIMULATION_H_ */
