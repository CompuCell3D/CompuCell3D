#ifndef LENGTHCONSTRAINTDATA_H
#define LENGTHCONSTRAINTDATA_H

#include "LengthConstraintDLLSpecifier.h"


namespace CompuCell3D {

    class LENGTHCONSTRAINT_EXPORT LengthConstraintData {

    public:
        LengthConstraintData() : lambdaLength(0.0), targetLength(0.0), minorTargetLength(0.0) {}

        double lambdaLength;
        double targetLength;
        double minorTargetLength;
    };

};
#endif
