#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <string>

namespace CompuCell3D {

    /*
     * Interface for Boundary Conditions
     */
    class Boundary {


    public:

        virtual bool applyCondition(int &coordinate, const int &max_value) = 0;

        virtual ~Boundary();


    };



};

#endif
