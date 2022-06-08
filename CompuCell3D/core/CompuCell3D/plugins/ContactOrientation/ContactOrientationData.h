
#ifndef CONTACTORIENTATIONPATA_H
#define CONTACTORIENTATIONPATA_H


#include <PublicUtilities/Vector3.h>
#include "ContactOrientationDLLSpecifier.h"

namespace CompuCell3D {


    class CONTACTORIENTATION_EXPORT ContactOrientationData {
    public:
        ContactOrientationData() {};

        ~ContactOrientationData() {};
        Vector3 oriantationVec;
        double alpha;


    };
};
#endif
