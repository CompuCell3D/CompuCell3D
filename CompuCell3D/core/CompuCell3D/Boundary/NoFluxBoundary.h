#ifndef NOFLUXBOUNDARY_H
#define NOFLUXBOUNDARY_H

#include "Boundary.h"


namespace CompuCell3D {

    /*
     * NoFluxBoundary. 
     */
    class NoFluxBoundary : public Boundary {


    public:

        bool applyCondition(int &coordinate, const int &max_value);

    };

};


#endif
