/**
 * LEAD University
 * Data Science Program
 * BCD-9218: Parallel and Distributed Computing
 * Instructor Diego Jimenez, Eng. (diego.jimenez@ulead.ac.cr)
 * OpenMP parallel Strassen algorithm for matrix multiplicationplication.
 */

#include <cstdio>
#include <cstdlib>
#include "timer.h"
#include "/usr/local/opt/libomp/include/omp.h"
#include <iostream>
#include "io.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;


void sum(vector<vector<int> > &A, vector<vector<int> > &B, vector<vector<int> > &C, int size)
{
    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtraction(vector<vector<int> > &A, vector<vector<int> > &B, vector<vector<int> > &C, int size)
{
    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void multiplication(vector<vector<int> > &A, vector<vector<int> > &B, vector<vector<int> > &C, int size)
{
    int i, j, k;
    for(i = 0; i < size; ++i)
        for(j = 0; j < size; ++j)
            for(k = 0; k < size; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
}


// TODO: Function implementing Strassen's algorithm
void strassen(int **A, int **B, int **C, int N) {

    // TODO: YOUR CODE GOES HERE
    int new_size = N / 2;
    vector<int> z(new_size);
    vector<vector<int> >
            a11(new_size, z), a12(new_size, z), a21(new_size, z), a22(new_size, z),
            b11(new_size, z), b12(new_size, z), b21(new_size, z), b22(new_size, z),
            c11(new_size, z), c12(new_size, z), c21(new_size, z), c22(new_size, z),
            M1(new_size, z), M2(new_size, z), M3(new_size, z), M4(new_size, z),
            M5(new_size, z), M6(new_size, z), M7(new_size, z),
            aResult(new_size, z), bResult(new_size, z);

    int i, j;

//dividing the matrices into subtraction-matrices:
    for (i = 0; i < new_size; i++) {
        for (j = 0; j < new_size; j++) {
            a11[i][j] = A[i][j];
            a12[i][j] = A[i][j + new_size];
            a21[i][j] = A[i + new_size][j];
            a22[i][j] = A[i + new_size][j + new_size];

            b11[i][j] = B[i][j];
            b12[i][j] = B[i][j + new_size];
            b21[i][j] = B[i + new_size][j];
            b22[i][j] = B[i + new_size][j + new_size];
        }
    }

#pragma omp parallel
    {
#pragma omp single
        {

            sum(a11, a22, aResult, new_size);
            sum(b11, b22, bResult, new_size);
            multiplication(aResult, bResult, M1, new_size);


            sum(a21, a22, aResult, new_size); // a21 + a22
            multiplication(aResult, b11, M2, new_size);
            // p2 = (a21+a22) * (b11)

            subtraction(b12, b22, bResult, new_size);      // b12 - b22
            multiplication(a11, bResult, M3, new_size);
            // p3 = (a11) * (b12 - b22)

            subtraction(b21, b11, bResult, new_size);       // b21 - b11
            multiplication(a22, bResult, M4, new_size);
            // p4 = (a22) * (b21 - b11)

            sum(a11, a12, aResult, new_size);      // a11 + a12
            multiplication(aResult, b22, M5, new_size);
            // p5 = (a11+a12) * (b22)

            subtraction(a21, a11, aResult, new_size);      // a21 - a11
            sum(b11, b12, bResult, new_size);
            // b11 + b12
            multiplication(aResult, bResult, M6, new_size);
            // p6 = (a21-a11) * (b11+b12)

            subtraction(a12, a22, aResult, new_size);      // a12 - a22
            sum(b21, b22, bResult, new_size);
            // b21 + b22
            multiplication(aResult, bResult, M7, new_size);
            // p7 = (a12-a22) * (b21+b22)
        }
    }

    // calculating c21, c21, c11 e c22:

    sum(M3, M5, c12, new_size); // c12 = p3 + p5
    sum(M2, M4, c21, new_size); // c21 = p2 + p4

    sum(M1, M4, aResult, new_size);       // p1 + p4
    sum(aResult, M7, bResult, new_size);  // p1 + p4 + p7
    subtraction(bResult, M5, c11, new_size); // c11 = p1 + p4 - p5 + p7

    sum(M1, M3, aResult, new_size);       // p1 + p3
    sum(aResult, M6, bResult, new_size);  // p1 + p3 + p6
    subtraction(bResult, M2, c22, new_size); // c22 = p1 + p3 - p2 + p6

    for (i = 0; i < new_size; i++)
    {
        for (j = 0; j < new_size; j++)
        {
            C[i][j] = c11[i][j];
            C[i][j + new_size] = c12[i][j];
            C[i + new_size][j] = c21[i][j];
            C[i + new_size][j + new_size] = c22[i][j];
        }
    }
}





// Main method
int main(int argc, char* argv[]) {
	int N;
	int **A, **B, **C;
	double elapsedTime;

	// checking parameters
	if (argc != 2 && argc != 4) {
		cout << "Parameters: <N> [<fileA> <fileB>]" << endl;
		return 1;
	}
	N = atoi(argv[1]);

	// allocating matrices
	A = new int*[N];
	B = new int*[N];
	C = new int*[N];
	for (int i=0; i<N; i++){
		A[i] = new int[N];
		B[i] = new int[N];
		C[i] = new int[N];
	}

	// reading files (optional)
	if(argc == 4){
		readMatrixFile(A,N,argv[2]);
		readMatrixFile(B,N,argv[3]);
	}

	// starting timer
	timerStart();

	// TODO: YOUR CODE GOES HERE


	// testing the results is correct
	if(argc == 4){
		printMatrix(C,N);
	}

	// stopping timer
	elapsedTime = timerStop();

	cout << "Duration: " << elapsedTime << " seconds" << std::endl;

	// releasing memory
	for (int i=0; i<N; i++) {
		delete [] A[i];
		delete [] B[i];
		delete [] C[i];
	}
	delete [] A;
	delete [] B;
	delete [] C;

	return 0;
}
