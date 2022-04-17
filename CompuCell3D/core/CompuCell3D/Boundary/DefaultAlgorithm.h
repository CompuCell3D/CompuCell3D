#ifndef DEFAULTALGORITHM_H
#define DEFAULTALGORITHM_H

#include "Algorithm.h"

namespace CompuCell3D {

    /*
     * Default Algorithm. 
     */
    class DefaultAlgorithm : public Algorithm {


    public:

        void readFile(const int index, const int size, string inputfile);

        bool inGrid(const Point3D &pt);

        int getNumPixels(int x, int y, int z);
    };
};


#endif
