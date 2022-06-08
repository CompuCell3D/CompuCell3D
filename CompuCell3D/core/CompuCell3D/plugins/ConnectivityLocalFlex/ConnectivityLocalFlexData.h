#ifndef CONNECTIVITYLOCALFLEXDATA_H
#define CONNECTIVITYLOCALFLEXDATA_H

#include "ConnectivityLocalFlexDLLSpecifier.h"

namespace CompuCell3D {

    class CONNECTIVITYLOCALFLEX_EXPORT ConnectivityLocalFlexData {

    public:
        ConnectivityLocalFlexData() : connectivityStrength(0.0) {}

        double connectivityStrength;


    };

};
#endif
