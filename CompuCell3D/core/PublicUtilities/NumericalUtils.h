#ifndef NUMERICALUTILS_H
#define NUMERICALUTILS_H

#include <Utils/Coordinates3D.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <vector>
#include <complex>



float findMin( float _d , int _dim );

double dist(double x0,double y0,double z0);
double dist(double x0,double y0,double z0,double x1,double y1,double z1);
//the takes as an input a vector 'a' of 4 real coefficient of cubic plynomial (a[0]*x^3 + a[1]*x^2+a[2]*x+a[3])
// and returns a vector of 3 complex solutions of the cubic equation
std::vector<std::complex<double> > solveCubicEquationRealCoeeficients(std::vector<double> & aRef);

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

double round(double number);

namespace CompuCell3D{


	class CellG;
	class Point3D;
	class BoundaryStrategy;

	class InertiaTensorComponents{
	public:
		InertiaTensorComponents():
			iXX(0.0),
				iYY(0.0),
				iZZ(0.0),
				iXY(0.0),
				iXZ(0.0),
				iYZ(0.0)
			{}
			double iXX;
			double iYY;
			double iZZ;
			double iXY;
			double iXZ;
			double iYZ;
	};

	Coordinates3D<double> precalculateCentroid(const Point3D & pt, const CellG *_cell, int _volumeIncrement,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy=0);

	double distInvariantCM(double x0,double y0,double z0,double x1,double y1,double z1,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy=0);
    
    double distanceInvariantCM(double x0,double y0,double z0,double x1,double y1,double z1,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy=0,int _pbcX=0,int _pbcY=0,int _pbcZ=0);
    
    
	Point3D distanceVectorInvariant(const Point3D & _pt1 ,const Point3D & _pt0,const Point3D & _fieldDim);
    
    Coordinates3D<double> distanceVectorCoordinatesInvariant(const Coordinates3D<double> & _pt1 ,const Coordinates3D<double> & _pt0,const Point3D & _fieldDim);
    

	std::pair<InertiaTensorComponents,InertiaTensorComponents> precalculateInertiaTensorComponentsAfterFlip(const Coordinates3D<double> & ptTrans,const CellG *newCell ,const CellG *oldCell);

	typedef std::pair<Coordinates3D<float>, Coordinates3D<float> > CenterOfMassPair_t;
	CenterOfMassPair_t precalculateAfterFlipCM(const Point3D &pt, const CellG *newCell, const CellG *oldCell,const Point3D & fieldDim, const Point3D & boundaryConditionIndicator);

	


};
#endif
