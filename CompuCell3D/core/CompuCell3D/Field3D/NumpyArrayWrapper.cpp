//
// Created by m on 10/5/24.
//

#include "NumpyArrayWrapper.h"
#include <iostream>

using namespace CompuCell3D;
using namespace std;

NumpyArrayWrapper::NumpyArrayWrapper(array_size_t dim_x, array_size_t dim_y){
    this->dim_x = dim_x;
    this->dim_y = dim_y;
    cerr<<"allocating array"<<endl;
    array.assign(dim_x*dim_y, 0.0);
    cerr<<"array size="<<array.size()<<endl;
};

void NumpyArrayWrapper::printArray() {

    for (array_size_t i = 0; i < dim_x; ++i){
        for (array_size_t j = 0; j < dim_y; ++j){
            cerr<<"a["<<i<<","<<j<<"]="<<array[i*dim_y+j]<<endl;
        }
    }
}



