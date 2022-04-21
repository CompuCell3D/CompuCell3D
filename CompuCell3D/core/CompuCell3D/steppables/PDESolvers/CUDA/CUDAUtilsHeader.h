#ifndef CUDAUTILSHEADER_H
#define CUDAUTILSHEADER_H

#include <stdio.h>

//TODO: check if we can use ASSERT_OR_THROW here


//#include <cutil_inline.h>
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        char buff[256];
        sprintf(buff, "%s(%i) : CUDA Runtime API error %d: %s", file, line, (int) err, cudaGetErrorString(err));
        fprintf(stderr, "%s\n", buff);
        //ASSERT_OR_THROW(buff, err==cudaSuccess);

        exit(-1);
    }
}

#endif
