#ifndef PERIODICBOUNDARY_H
#define PERIODICBOUNDARY_H

#include "Boundary.h"


namespace CompuCell3D {

    /*
     * PeriodicBoundary class
     */
    class PeriodicBoundary : public Boundary {


    public:

        bool applyCondition(int &coordinate, const int &max_value);


    };

};

#endif
