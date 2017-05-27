/*
 * CudaUtilities.cuh
 *
 *  Created on: May 27, 2017
 *      Author: rostifar
 */

#ifndef CUDAUTILITIES_CUH_
#define CUDAUTILITIES_CUH_

void *fixedCudaMalloc(size_t len);

template<typename FN>
void cudaAlloCopy(FN *org, FN *res, const size_t size);

#endif /* CUDAUTILITIES_CUH_ */
