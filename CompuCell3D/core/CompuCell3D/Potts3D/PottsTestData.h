#ifndef POTTSTESTDATA_H
#define POTTSTESTDATA_H

#include <CompuCell3D/Field3D/Point3D.h>


namespace CompuCell3D {

    class PottsTestData {

    public:
        PottsTestData(){}

        Point3D changePixel;
        Point3D changePixelNeighbor;
        double motility;
        bool pixelCopyAccepted;
        



    };

};


#endif

