/*
 * hTypes.h
 *
 *  Created on: Jun 1, 2017
 *      Author: rostifar
 */

#ifndef HTYPES_H_
#define HTYPES_H_

#include <stdio.h>
#include <type_traits>
#include <stdexcept>

namespace Helix {

template<typename F>
class Vector2 {
public:
	F x, y;
	const auto len = 2;
	typedef Vector2<F> vec2;

	Vector2() {
		x = 0;
		y = 0;
	}

	Vector2(F a = 0, F b = 0) {
		x = a;
		y = b;
	}

	F operator [](int i) const {
		switch (i) {
			case 0:
				return x;
			case 1:
				return y;
			default:
				throw std::out_of_range("Error, index of range. Type Vector2 is only indexable via 0, 1");
		}
	}

	vec2 operator + (vec2 a) const {
		return vec2(x + a.x, y + a.y);
	}

	vec2 operator - (vec2 a) const {
		return vec2(x - a.x, y - a.y);
	}

	void scale(F a) {
		x *= a;
		y *= a;
	}

	F dot(vec2 v) {
		return (x * v.x + y * v.y);
	}

	void print() {
		if(std::is_same<F, float>::value) {
			printf("(%f, %f, %f)\n", x, y);
		} else {
			printf("(%lf, %lf, %lf)\n", x, y);
		}
	}

	F *map() {
		F *arr = new F[2];
		for (int i = 0; i < len; i++) {
			arr[i] = this[i];
		}
		return arr;
	}

	void map(F *arr, int i) {
		for (auto q = 0; i < len; q++, i++) {
			arr[i] = this[q];
		}
	}
};

template<typename F>
class Vector3 {
public:
	F x, y, z;
	const auto len = 3;
	typedef Vector3<F> vec3;

	Vector3() {
		x = 0;
		y = 0;
		z = 0;
	}

	Vector3(F _x) {
		x = _x;
		y = 0;
		z = 0;
	}

	Vector3(F _x, F _y) {
		x = _x;
		y = _y;
		z = 0;
	}

	Vector3(F _x, F _y, F _z) {
		x = _x;
		y = _y;
		z = _z;
	}

	Vector3(Vector2<F> a, F b) {
		x = a.x;
		y = a.y;
		z = b;
	}

	F operator [](int i) const {
		switch (i) {
			case 0:
				return x;
			case 1:
				return y;
			default:
				return z;
		}
	}

	F & operator [](int i) {
		switch (i) {
			case 0:
				return x;
			case 1:
				return y;
			default:
				return z;
		}
	}

	vec3 operator + (vec3 v) const {
		F xr = x + v.x;
		F yr = y + v.y;
		F zr = z + v.z;
		return vec3(xr, yr, zr);
	}

	vec3 operator - (vec3 v) const {
		F xr = x - v.x;
		F yr = y - v.y;
		F zr = z - v.z;
		return vec3(xr, yr, zr);
	}

	void scale(F s) {
		x *= s;
		y *= s;
		z *= s;
	}

	F dot(vec3 v) {
		return (x * v.x + y * v.y + z * v.z);
	}

	vec3 cross(vec3 v) {
		vec3 c;
		c[0] = y * v.z - z * v.y;
		c[1] = z * v.x - x * v.z;
		c[2] = x * v.y - y * v.x;
		return c;
	}

	void normalize() {
		F s = sqrt(x * x + y * y + z * z);
		x /= s;
		y /= s;
		z /= s;
	}

	Vector normalized() {
		Vector<F> n;
		F s = sqrt(x * x + y * y + z * z);
		x /= s;
		y /= s;
		z /= s;
		return vec3(x, y, z);
	}

	void print() {
		if(std::is_same<F, float>::value) {
			printf("(%f, %f, %f)\n", x, y, z);
		} else {
			printf("(%lf, %lf, %lf)\n", x, y, z);
		}
	}

	F *map() {
		F *arr = new F[3];
		for (int i = 0; i < len; i++) {
			arr[i] = this[i];
		}
		return arr;
	}

	void map(F *arr, int i) {
		for (auto q = 0; i < len; q++, i++) {
			arr[i] = this[q];
		}
	}
};

template<typename F>
class Vector4 {
public:
	F x, y, z, w;
	auto const len = 4;

	Vector4() {
		x = 0;
		y = 0;
		z = 0;
		w = 0;
	}

	Vector4(F _x = 0, F _y = 0, F _z = 0, F _w = 0) {
		x = _x;
		y = _y;
		z = _z;
		w = _w;
	}

	Vector4(Vector2<F> a, Vector2<F> b) {
		x = a.x;
		y = a.y;
		z = b.x;
		w = b.y;
	}

	Vector4(Vector3<F> a, F b) {
		x = a.x;
		y = a.y;
		z = a.z;
		w = b;
	}

	F operator[](int i) {
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
				throw std::out_of_range("Error, index of range. Type Vector4 is only indexable via 0, 1, 2, 3");
		}
	}

	F *map() {
		F *arr = new F[4];
		for (int i = 0; i < len; i++) {
			arr[i] = this[i];
		}
		return arr;
	}

	void map(F *arr, int i) {
		for (auto q = 0; i < len; q++, i++) {
			arr[i] = this[q];
		}
	}
};

}

#endif /* HTYPES_H_ */
