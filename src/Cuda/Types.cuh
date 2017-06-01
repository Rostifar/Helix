/*
 * Types.cuh
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */

#ifndef TYPES_CUH_
#define TYPES_CUH_
#include <stdexcept>
#include <vector>

namespace Helix {

template<typename F>
struct F2 {
	F x, y;
	F operator[](int i);
	static F *pkg(F2<F> *a);
	static int map(F *arr, F2<F> *a, int i);
	static F2<F> fuse(F a, F b);
};

template<typename F>
F F2<F>::operator [](int i) {
	switch (i) {
		case 0:
			return x;
		case 1:
			return y;
		default:
			throw std::out_of_range("Error! Indexing out of range. F2 can only be indexed by 0, 1");
	}
}

template<typename F>
F *F2<F>::pkg(F2<F> *a) {
	F *arr = new F[2];
	arr[0] = a->x; arr[1] = a->y;
	return arr;
}

template<typename F>
int F2<F>::map(F *arr, F2<F> *a, int i) {
	int len = sizeof(a) / sizeof(F2<F>);
	if (len == 1) {
		arr[i] = a->x; arr[i + 1] = a->y;
	}
	else {
		for (int q = 0; q < len; q++) {
			arr[i] = a[q].x; ++i;
			arr[i] = a[q].y; ++i;
		}
	}
	return i;
}

template<typename F>
F2<F> fuse(F a, F b) {
	return makeF2(a, b);
}



template<typename F>
struct F3 {
	F x, y, z;
	F operator[](int i);
	static F *pkg(F3<F> *a);
	static int map(F *arr, F3<F> *a, int i);
	static F3<F> fuse(F2<F> *a, F b);
};

template<typename F>
F F3<F>::operator[](int i) {
	switch (i) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			throw std::out_of_range("Error! Indexing out of range. F3 can only be indexed by 0, 1, 2");
	}
}

template<typename F>
F *F3<F>::pkg(F3<F> *a) {
	F *arr = new F[3];
	arr[0] = x; arr[1] = y; arr[2] = z;
	return arr;
}

template<typename F>
int F3<F>::map(F *arr, F3<F> *a, int i) {
	int len = sizeof(a) / sizeof(F3<F>);
	if (len == 1) {
		arr[i] = a->x; ++i;
		arr[i] = a->y; ++i;
		arr[i] = a->z; ++i;
	}
	else {
		for (int q = 0; q < len; q++) {
			arr[i] = a[q].x; ++i;
			arr[i] = a[q].y; ++i;
			arr[i] = a[q].z; ++i;
		}
	}
	return i;
}

template<typename F>
F3<F> F3<F>::fuse(F2<F> *a, F b)  {
	return makeF3<F>(a->x, a->y, b);
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
