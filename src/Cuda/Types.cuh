/*
 * Types.cuh
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */

#ifndef TYPES_CUH_
#define TYPES_CUH_

namespace Helix {

/* @deprecated
template<typename F>
struct FN {
	F   *vec;
	int len;

	FN(int n) {
		len = n;
		vec = malloc(sizeof(F) * n);
	}
	~FN() {
		delete vec; //stack call
	}
	F operator[](const int i) {
		return vec[i];
	}
	void operator=(F *value) {
		vec = value;
	}
};
*/

template<typename F>
struct F2 {
	F x, y;
};

template<typename F>
struct F3 {
	F x, y, z;
	int map(F *arr, int i);
	static F3<F> fuse(F2<F> a, F b);
};

template<typename F>
int F3<F>::map(F *arr, int i) {
	arr[i] = x; arr[i + 1] = y; arr[i + 2] = z;
	return i += 3;
}

template<typename F>
static F3<F> fuse(F2<F> a, F b)  {
	return makeF3<F>(a.x, a.y, b);
}

template<typename F>
struct F4 {
	F x, y, z, w;
	void map(F *arr, int i);
	static F4<F> fuse(F2<F> a, F2<F> b);
};

template<typename F>
void F4<F>::map(F *arr, int i) {
	arr[i] = x; arr[i + 1] = y; arr[i + 2] = z; arr[i + 3] = w;
	return i += 4;
}

template<typename F>
F4<F> F4<F>::fuse(F2<F> a, F2<F> b) {
	return makeF4(a.x, a.y, b.x, b.y);
}

template<typename F>
struct GenerationLimits {
	F3<F> *vec;
	int   *strides;

	GenerationLimits(int components, F *ceilings, F *floors, F *distribTypes, int *_strides) : vec(components) {
		strides = _strides;
		for (int i = 0, q = 0; i < components; i++) {
			vec[q]   = ceilings    [i];
			vec[++q] = floors  	   [i];
			vec[++q] = distribTypes[i];
		}
	}

	GenerationLimits(F *_vec, int *_strides) {
		vec     = _vec;
		strides = _strides;
	}

	~GenerationLimits() {
		delete strides;
	}
};

template<typename _F>
struct UniverseParams {
	int epsilon, dt, particles, partitions, epochs, offset;
	typedef _F F;
	UniverseParams(int _ep, int _dt, int _particles, int _partitions, int _epochs, int _offset)
		: epsilon(_ep), dt(_dt), particles(_particles), partitions(_partitions), epochs(_epochs), offset(_offset){};
};

}


#endif /* TYPES_CUH_ */
