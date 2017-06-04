#include "../Vector.h"
#include "../Matrix.h"

template<typename F>
class Camera {
public:

	Vector<F> position;
	Vector<F> gaze;
	Vector<F> up;

	F xFov; // Measured in degrees
	F yFov;
	long nRows;
	long nCols;

	F maxClipping;

	void setMinClipping(int kernelSize, F particleRadius) {
		F xMaxFov = ((F) kernelSize / (F) nCols) * xFov;
		F yMaxFov = ((F) kernelSize / (F) nRows) * yFov;
		F maxFov = (xMaxFov < yMaxFov) ? xMaxFov : yMaxFov;
		minClipping = particleRadius / tan((M_PI / 180) * (0.5 * maxFov));
	} F getMinClipping() {return minClipping;}

	Matrix<F> getTranslationMatrix() {

		Matrix<F> M(4, 4);
		M(0, 0) = 1; M(0, 1) = 0; M(0, 2) = 0; M(0, 3) = -position.x;
		M(1, 0) = 0; M(1, 1) = 1; M(1, 2) = 0; M(1, 3) = -position.y;
		M(2, 0) = 0; M(2, 1) = 0; M(2, 2) = 1; M(2, 3) = -position.z;
		M(3, 0) = 0; M(3, 1) = 0; M(3, 2) = 0; M(3, 3) = 1;
		return M;
	}

	Matrix<F> getViewMatrix() {
		Vector<F> xAxis, yAxis, zAxis;

		zAxis = gaze.normalized();
		xAxis = up.cross(zAxis).normalized();
		yAxis = zAxis.cross(xAxis);

		Matrix<F> M(4, 4);
		M(0, 0) = xAxis.x; M(0, 1) = yAxis.x; M(0, 2) = zAxis.x; M(0, 3) = 0;
		M(1, 0) = xAxis.y; M(1, 1) = yAxis.y; M(1, 2) = zAxis.y; M(1, 3) = 0;
		M(2, 0) = xAxis.z; M(2, 1) = yAxis.z; M(2, 2) = zAxis.z; M(2, 3) = 0;
		M(3, 0) = 0;       M(3, 1) = 0;       M(3, 2) = 0;       M(3, 3) = 1;
		Matrix<F> T = getTranslationMatrix();
		return (M * T);
	}

	Matrix<F> getPerspectiveMatrix() {

	   /*
		* Dimensions of the frustum:
		*
		* n: near z view perspective distance
		* f: far z view perspective distance
		* t: top of the frustum at the near plane
		* b: bottom of the frustum at the near plane
		* r: right of the frustum at the near plane
		* l: left of the frustum at the near plane
		*
		* This model assumes that the frustum is centered on the z-axis
		* In the future, the preceding parameters could be changed to create
		* asymmetry
		*
		* Note: x / y scaling is completely dependent on the view perspective
		* z value. Therefore; in a symmetrical frustum, all particles will
		* project a circle onto the screen
	    */

		F n = minClipping;
		F f = maxClipping;
		F t = n * tan((M_PI / 180) * (yFov / 2));
		F b = -t;
		F r = n * tan((M_PI / 180) * (xFov / 2));
		F l = -r;


		Matrix<F> M(4, 4);
		M(0, 0) =  (2 * n) / (r - l);  M(0, 1) = 0;
		M(1, 0) = 0;                   M(1, 1) = (2 * n) / (t - b);
		M(2, 0) = 0;                   M(2, 1) = 0;
		M(3, 0) = 0;                   M(3, 1) = 0;

		M(0, 2) =  (r + l) / (r - l);  M(0, 3) = 0;
		M(1, 2) =  (t + b) / (t - b);  M(1, 3) = 0;
		M(2, 2) = (-f - n) / (f - n);  M(2, 3) = (-2 * f * n) / (f - n);
		M(3, 2) = -1;                  M(3, 3) = 0;
		Matrix<F> V = getViewMatrix();
		return (M * V);
	}

private:

	F minClipping;
};
