#ifndef CABOUNDARYCONDITIONSPECIFIER_H
#define CABOUNDARYCONDITIONSPECIFIER_H

#include "CAPDESolversDLLSpecifier.h"
namespace CompuCell3D {

class CAPDESOLVERS_EXPORT CABoundaryConditionSpecifier{
public:
	enum BCType{
		PERIODIC,
		CONSTANT_VALUE,
		CONSTANT_DERIVATIVE
	};

	enum BCPosition{INTERNAL=-2,BOUNDARY,
		MIN_X, MAX_X,
		MIN_Y, MAX_Y,
		MIN_Z, MAX_Z
	};

	CABoundaryConditionSpecifier(){
		planePositions[0]=CONSTANT_DERIVATIVE;//min X
		planePositions[1]=CONSTANT_DERIVATIVE;//max X
		planePositions[2]=CONSTANT_DERIVATIVE;//min Y
		planePositions[3]=CONSTANT_DERIVATIVE;//max Y
		planePositions[4]=CONSTANT_DERIVATIVE;//min Z
		planePositions[5]=CONSTANT_DERIVATIVE;//max Z

		//planePositions[0]=2;//min X
		//planePositions[1]=2;//max X
		//planePositions[2]=2;//min Y
		//planePositions[3]=2;//max Y
		//planePositions[4]=2;//min Z
		//planePositions[5]=2;//max Z

        values[0]=0.0;
        values[0]=0.0;
        values[1]=0.0;
        values[2]=0.0;
        values[3]=0.0;
        values[4]=0.0;
        values[5]=0.0;
	}

    void setPlanePosition(unsigned int pos, BCType val){planePositions[pos]=val;}
    void setValues(unsigned int pos, double val){values [pos]=val;}

	BCType planePositions[6];
	//int planePositions[6];
	double  values [6];


};

}

#endif