#ifndef CONTACTLOCALPRODUCTDATA_H
#define CONTACTLOCALPRODUCTDATA_H

#include "ContactLocalProductDLLSpecifier.h"
#include <vector>

namespace CompuCell3D {

    class CellG;


    class CONTACTLOCALPRODUCT_EXPORT ContactLocalProductData {

    public:
        typedef std::vector<float> ContainerType_t;

        ContactLocalProductData() : jVec(std::vector<float>(2, 0.0)) {}

        std::vector<float> jVec;

    };


};
#endif
