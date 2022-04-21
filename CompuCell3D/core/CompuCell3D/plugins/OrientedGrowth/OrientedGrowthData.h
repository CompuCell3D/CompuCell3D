
#ifndef ORIENTEDGROWTHPATA_H
#define ORIENTEDGROWTHPATA_H


#include <vector>
#include "OrientedGrowthDLLSpecifier.h"

namespace CompuCell3D {


    class ORIENTEDGROWTH_EXPORT OrientedGrowthData {
    public:
        OrientedGrowthData() {};

        ~OrientedGrowthData() {};
        std::vector<float> array;
        int x;
        float elong_x;
        float elong_y;
        float elong_targetWidth;
        bool elong_enabled;
    };
};
#endif
