#ifndef CONNECTIVITYGLOBALDATA_H
#define CONNECTIVITYGLOBALDATA_H


#include "ConnectivityGlobalDLLSpecifier.h"

namespace CompuCell3D {

    class CONNECTIVITYGLOBAL_EXPORT ConnectivityGlobalData {

    public:
        ConnectivityGlobalData() : connectivityStrength(0.0) {}

        double connectivityStrength;


    };

};
#endif
