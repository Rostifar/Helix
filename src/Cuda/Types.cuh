/*
 * Types.cuh
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */

#ifndef TYPES_CUH_
#define TYPES_CUH_
#include <stdexcept>

namespace Helix {

template<typename F>
struct F2 {
	F x, y;
	static inline void	segment(F *arr, F2<F> a, int i);
	static void			interSegment(F *arr, F2<F> *a, int i, int ceiling);
};

template<typename F>
void F2<F>::segment(F *arr, F2<F> a, int i) {
	arr[i] = a.x; arr[i + 1].y;
}

template<typename F>
void F2<F>::interSegment(F *arr, F2<F> *a, int i, int ceiling) {
	for (int q = 0; i < ceiling; q++) {
		arr[i]	= a[q].x; ++i;
		arr[i]	= a[q].y; ++i;
	}
}


template<typename F>
struct F3 {
	F x, y, z;
	int map(F *arr, int i);
	static F3<F> fuse   (F2<F> a, F b);
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
	F operator[](int i);
	void map(F *arr, int i);
	static F4<F> fuse(F2<F> a, F2<F> b);
	static inline void segment(F *arr, F4<F> a, int i);
};

template<typename F>
F F4<F>::operator[](int i) {
	switch (i) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		case 3:
			return w;
		default:
			throw std::out_of_range("The F4 type only has 4 values!");
	}
}

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
void F4<F>::segment(F *arr, F4<F> a, int i) {
	for (int q; q < 4; q++, i++) arr[i] = a[q];
}

/* @deprecated
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
*/

}


#endif /* TYPES_CUH_ */
