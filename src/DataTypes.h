/*
 * DataTypes.h
 *
 *  Created on: May 29, 2017
 *      Author: rostifar
 */

#ifndef DATATYPES_H_
#define DATATYPES_H_

namespace Helix {

enum DataFmt {
	SEGMENTED,
	UNIFORM
};

template<typename F>
struct FN {
	F *vec;
	int len;

	FN(int n) {
		len = n;
		vec = malloc(sizeof(F) * n);
	}
	F operator[](const int i) {
		return vec[i];
	}
	void operator=(F *value) {
		vec = value;
	}
	size_t size(int j) {
		return sizeof(vec);
	}
};

}

#endif /* DATATYPES_H_ */
