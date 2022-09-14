#ifndef LENGTHCONSTRAINTLOCALFLEXDATA_H
#define LENGTHCONSTRAINTLOCALFLEXDATA_H

#include "LengthConstraintLocalFlexDLLSpecifier.h"


namespace CompuCell3D {

    class LENGTHCONSTRAINTLOCALFLEX_EXPORT LengthConstraintLocalFlexData {

    public:
        LengthConstraintLocalFlexData() : lambdaLength(0.0), targetLength(0.0) {}

        double lambdaLength;
        double targetLength;


    };

};
#endif
