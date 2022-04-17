#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <string>

using namespace std;

#include <CompuCell3D/Field3D/Dim3D.h>

namespace CompuCell3D {
    class Point3D;

    /*
      * Interface for Algorithm
      */
    class Algorithm {


    public:
        virtual ~Algorithm() {}

        virtual void readFile(const int index, const int size, string inputfile) = 0;

        virtual bool inGrid(const Point3D &pt) = 0;

        void setDim(Dim3D theDim) { dim = theDim; }

        void setCurrentStep(int theCurrentStep) { currentStep = theCurrentStep; }

        virtual int getNumPixels(int x, int y, int z) = 0;

    protected:
        Dim3D dim;
        int currentStep;
    };

};

#endif
