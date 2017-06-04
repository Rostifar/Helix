#include "Camera.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

const long MAX_THREAD_SIZE = 1024;

template<typename F>
void render(Camera<F> camera, F* particleCoordinates, long nParticles);
