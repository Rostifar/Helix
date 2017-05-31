/*
 * UniverseSimulation.h
 *
 *  Created on: May 30, 2017
 *      Author: rostifar
 */
#include "../Types.cuh"
#include <string>

#ifndef UNIVERSESIMULATION_H_
#define UNIVERSESIMULATION_H_

namespace Helix {

template<typename F>
struct UniParticle {
	F3<F>  r;
	F3<F>  velocity;
	F3<F>  acceleration;
	F	   mass;
	size_t size = sizeof(F3<F>) * 3 + sizeof(F);

	const static int len  = 10;

	UniParticle(F3<F> _r, F3<F> _velocity, F3<F> _acc, F _mass);
	static F *toCudaFmt(UniParticle<F> *particles);
	static UniParticle<F> *toParticleFmt(F *arr);
};

template<typename F>
UniParticle<F>::UniParticle(F3<F> _r, F3<F> _velocity, F3<F> _acc, F _mass) {
	r 			 = _r;
	velocity 	 = _velocity;
	acceleration = _acc;
	mass 		 = _mass;
}

template<typename F>
F *UniParticle<F>::toCudaFmt(UniParticle<F> *particles) {
	F *arr = malloc(sizeof(F) * (sizeof(particles) / sizeof(UniParticle)) * len);
	for (int i = 0, j = 0; sizeof(particles) / sizeof(UniParticle); i++) {
		j = r.map<F>(arr, j);
		j = velocity.map<F>(arr, j);
		j = acceleration.map<F>(arr, j);
		arr[j] = mass;
	}
	return arr;
}

template<typename F>
UniParticle<F> *toParticleFmt(F *arr) {
	UniParticle<F> *particles = malloc(sizeof(arr));
	for (int i = 0, q = 0; i < (sizeof(arr) / sizeof(F)) / particles->len; i++) {
		particles[i].r 		      = makeF3(arr[q], arr[q + 1], arr[q + 2]); q += 3;
		particles[i].velocity     = makeF3(arr[q], arr[q + 1], arr[q + 3]); q += 3;
		particles[i].acceleration = makeF3(arr[q], arr[q + 1], arr[q + 3]); q += 3;
		particles[i].mass 		  = arr[q];
	}
	return particles;
}

template<typename F>
struct UniLimitFmt {
	F2<F> rLim;
	F2<F> vLim;
	F2<F> aLim;
	F2<F> mLim;
	F4<F> generationType;
	F4<F> *toCudaFmt();
};

template<typename F>
F4<F> *UniLimitFmt<F>::toCudaFmt() {
		F4<F> *fmt = new F4<F>[3];
		fmt[0] = fuse<F>(rLim, vLim);
		fmt[1] = fuse<F>(aLim, mLim);
		fmt[2] = generationType;
		return fmt;
	}

template<typename F>
class UniverseSimulation {
public:
	UniverseSimulation(int _nParticles, int _epsilon, F _dt);
	void addGenerationLimits(UniLimitFmt<F> *limits);
	void beginUniverseSimulation();
	void resumeUniverseSimulation(std::string file);
	void onEpochCompletion(UniParticle<F> *particles);
	virtual ~UniverseSimulation();

private:
	int            n;
	int            epsilon;
	int            epochs 		   = 0;
	F   		   dt;
	UniLimitFmt<F> limits;
	bool		   limitsSet 	   = false;
	Platform       genPlatform 	   = Platform::GPU;
	Platform       computePlatform = Platform::CPU;
	void pregenerateLimits();
};

template<typename F>
UniverseSimulation<F>::UniverseSimulation(int _nParticles, int _epsilon, F _dt) {
	n 			= nParticles;
	epsilon 	= _epsilon;
	dt 			= _dt;
}

template<typename F>
void UniverseSimulation<F>::addGenerationLimits(UniLimitFmt<F> *_limits) {
	limits = _limits;
	limitsSet = true;
}

template<typename F>
void UniverseSimulation<F>::pregenerateLimits() {

}

template<typename F>
void UniverseSimulation<F>::beginUniverseSimulation() {
	if (!limitsSet) pregenerateLimits();
}

template<typename F>
void UniverseSimulation<F>::onEpochCompletion(UniParticle<F> *particles) {

}

template<typename F>
void UniverseSimulation<F>::resumeUniverseSimulation(std::string file) {

}

}

#endif /* UNIVERSESIMULATION_H_ */
