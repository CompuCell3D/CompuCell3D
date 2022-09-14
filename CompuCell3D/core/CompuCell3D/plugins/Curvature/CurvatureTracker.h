
#ifndef CURVATURETRACKER_H
#define CURVATURETRACKER_H


/**
@author m
*/
#include <set>

#include "CurvatureDLLSpecifier.h"


namespace CompuCell3D {

    class CellG;


    class CURVATURE_EXPORT CurvatureTrackerData {
    public:

        CurvatureTrackerData(CellG *_neighborAddress = 0, float _lambdaCurvature = 0.0, float _activationEnergy = 0.0,
                             int _maxNumberOfJunctions = 100, int _neighborOrder = 1)
                : neighborAddress(_neighborAddress), lambdaCurvature(_lambdaCurvature),
                  activationEnergy(_activationEnergy), maxNumberOfJunctions(_maxNumberOfJunctions),
                  neighborOrder(_neighborOrder) {}

        CurvatureTrackerData(const CurvatureTrackerData &ctd) //copy constructor
        {
            neighborAddress = ctd.neighborAddress;
            lambdaCurvature = ctd.lambdaCurvature;
            activationEnergy = ctd.activationEnergy;
            maxNumberOfJunctions = ctd.maxNumberOfJunctions;
            neighborOrder = ctd.neighborOrder;


        }

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const CurvatureTrackerData &_rhs) const {
            return neighborAddress < _rhs.neighborAddress;
        }


        ///members
        CellG *neighborAddress;
        float lambdaCurvature;
        float activationEnergy;
        int maxNumberOfJunctions;
        int neighborOrder;

    };

    class CURVATURE_EXPORT CurvatureTracker {
    public:
        CurvatureTracker() {};

        ~CurvatureTracker() {};
        //stores ptrs to cell neighbors within a given cluster i.e. each cell keeps track of its neighbors
        std::set <CurvatureTrackerData> internalCurvatureNeighbors;
        CurvatureTrackerData ctd;
    };
};
#endif

