
template<typename F>
class Matrix {
public:

	Matrix(const Matrix &obj) {
		nRows = obj.nRows;
		nCols = obj.nCols;
		data = obj.data;
	}

	Matrix(F n, F m) {
		nRows = n;
		nCols = m;
		
		data = new F*[nRows];
		for (int i = 0; i < nRows; i++) {
			data[i] = new F[nCols];
		}

		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nCols; j++) {
				data[i][j] = 0.0;
			}
		}
	}

	int nRows;
	int nCols;
	
	F& operator()(int row, int col) {
		return data[row][col];
	}

	const F& operator()(int row, int col) const {
		return data[row][col];
	}

	Matrix<F> operator * (Matrix mat) const {
		Matrix<F> p(nRows, mat.nCols);

		for (int i = 0; i < mat.nCols; i++) {
			for (int j = 0; j < nRows; j++) {
				for (int k = 0; k < nCols; k++) {
					p(j, i) += data[j][k] *  mat(k, i);
				}
			}
		}
		return p;
	}

	void scale(F s) {
		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nCols; j++) {
				data[i][j] *= s;
			}
		}
	}

	Matrix<F> scaled(F s) {
		Matrix<F> p(nRows, nCols);
		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nCols; j++) {
				p(i, j) = s * data[i][j];
			}
		}
		return p;
	}

	void print(const char* s) {
		printf("%s:\n", s);
		print();
	}

	void print() {
		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nCols; j++) {
				printf("%lf ", data[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	void toArray(F* array) {
		for (int i = 0; i < nRows; i++) {
			for (int j = 0; j < nCols; j++) {
				int k = j + i * nCols;
				array[k] = data[i][j];
			}
		}
	}

private:

	F** data;
};
