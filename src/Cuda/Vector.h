#include <stdio.h>

template<typename F>
class Vector {
public:

	Vector() {
		x = 0;
		y = 0;
		z = 0;
	}

	Vector(F _x) {
		x = _x;
		y = 0;
		z = 0;
	};

	Vector(F _x, F _y) {
		x = _x;
		y = _y;
		z = 0;
	};

	Vector(F _x, F _y, F _z) {
		x = _x;
		y = _y;
		z = _z;
	};

	F x;
	F y;
	F z;

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

	Vector operator + (Vector v) const {
		F xr = x + v.x;
		F yr = y + v.y;
		F zr = z + v.z;
		return Vector(xr, yr, zr);
	}

	Vector operator - (Vector v) const {
		F xr = x - v.x;
		F yr = y - v.y;
		F zr = z - v.z;
		return Vector(xr, yr, zr);
	}

	void scale(F s) {
		x *= s;
		y *= s;
		z *= s;
	}

	F dot(Vector v) {
		return (x * v.x + y * v.y + z * v.z);
	}

	Vector cross(Vector v) {
		Vector<F> c;
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
		return Vector(x, y, z);
	}

	void print() {
		printf("(%lf, %lf, %lf)\n", x, y, z);
	}
};
