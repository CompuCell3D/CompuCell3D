#ifndef FOCALPOINTPLASTICITYTRACKER_H
#define FOCALPOINTPLASTICITYTRACKER_H


/**
@author m
*/
#include <set>

#include "FocalPointPlasticityDLLSpecifier.h"
#include <vector>
#include <CompuCell3D/Potts3D/Cell.h>

namespace CompuCell3D {

    class CellG;

    class FocalPointPlasticityLinkTrackerData;

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityTrackerData {
    public:

        FocalPointPlasticityTrackerData(CellG *_neighborAddress = 0, float _lambdaDistance = 0.0,
                                        float _targetDistance = 0.0, float _maxDistance = 100000.0,
                                        int _maxNumberOfJunctions = 0, float _activationEnergy = 0.0,
                                        int _neighborOrder = 1, bool _isInitiator = true, int _initMCS = 0)
                : neighborAddress(_neighborAddress), lambdaDistance(_lambdaDistance), targetDistance(_targetDistance),
                  maxDistance(_maxDistance), maxNumberOfJunctions(_maxNumberOfJunctions),
                  activationEnergy(_activationEnergy), neighborOrder(_neighborOrder), anchor(false), anchorId(0),
                  isInitiator(_isInitiator), initMCS(_initMCS) {

            anchorPoint = std::vector<float>(3, 0.);
        }

        FocalPointPlasticityTrackerData(const FocalPointPlasticityTrackerData &fpptd) //copy constructor
        {
            neighborAddress = fpptd.neighborAddress;
            lambdaDistance = fpptd.lambdaDistance;
            targetDistance = fpptd.targetDistance;
            maxDistance = fpptd.maxDistance;
            activationEnergy = fpptd.activationEnergy;
            maxNumberOfJunctions = fpptd.maxNumberOfJunctions;
            neighborOrder = fpptd.neighborOrder;
            anchor = fpptd.anchor;
            anchorId = fpptd.anchorId;
            anchorPoint = fpptd.anchorPoint;
            isInitiator = fpptd.isInitiator;
            initMCS = fpptd.initMCS;

        }

        // type cast from FocalPointPlasticityLinkTrackerData
        // be sure to set isInitiator after type cast!
        FocalPointPlasticityTrackerData(const FocalPointPlasticityLinkTrackerData &fppltd);

        FocalPointPlasticityTrackerData operator=(const FocalPointPlasticityLinkTrackerData &fppltd) {
            return FocalPointPlasticityTrackerData(fppltd);
        }

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const FocalPointPlasticityTrackerData &_rhs) const {
            // notice that anchor links will be listed first and the last of such links will have highest anchorId
            return neighborAddress < _rhs.neighborAddress ||
                   (!(_rhs.neighborAddress < neighborAddress) && anchorId < _rhs.anchorId);
        }


        ///members
        CellG *neighborAddress;
        float lambdaDistance;
        float targetDistance;
        float maxDistance;
        int maxNumberOfJunctions;
        float activationEnergy;
        int neighborOrder;
        bool anchor;
        std::vector<float> anchorPoint;
        bool isInitiator;
        int initMCS;

        int anchorId;

    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityTracker {
    public:
        FocalPointPlasticityTracker() {};

        ~FocalPointPlasticityTracker() {};
        std::set <FocalPointPlasticityTrackerData> focalPointPlasticityNeighbors; //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors
        std::set <FocalPointPlasticityTrackerData> internalFocalPointPlasticityNeighbors; //stores ptrs to cell neighbors withon a given cluster i.e. each cell keeps track of its neighbors

        std::set <FocalPointPlasticityTrackerData> anchors;

        FocalPointPlasticityTrackerData fpptd;
    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityJunctionCounter {
        unsigned char type;
    public:
        FocalPointPlasticityJunctionCounter(unsigned char _type) {
            type = _type;
        }

        bool operator()(const FocalPointPlasticityTrackerData &_fpptd) {
            return _fpptd.neighborAddress->type == type;
        }

    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityLinkTrackerData {
    public:

        // Tracker data associated with link
        FocalPointPlasticityLinkTrackerData(float _lambdaDistance = 0.0, float _targetDistance = 0.0,
                                            float _maxDistance = 100000.0, int _initMCS = 0)
                : lambdaDistance(_lambdaDistance), targetDistance(_targetDistance), maxDistance(_maxDistance),
                  anchor(false), anchorId(0), initMCS(_initMCS) {

            maxNumberOfJunctions = 0;
            activationEnergy = 0.;
            neighborOrder = 0;

            anchorPoint = std::vector<float>(3, 0.);
        }

        // type cast from FocalPointPlasticityTrackerData
        FocalPointPlasticityLinkTrackerData(const FocalPointPlasticityTrackerData &fpptd) {
            lambdaDistance = fpptd.lambdaDistance;
            targetDistance = fpptd.targetDistance;
            maxDistance = fpptd.maxDistance;
            anchor = fpptd.anchor;
            anchorId = fpptd.anchorId;
            anchorPoint = fpptd.anchorPoint;
            initMCS = fpptd.initMCS;

            maxNumberOfJunctions = fpptd.maxNumberOfJunctions;
            activationEnergy = fpptd.activationEnergy;
            neighborOrder = fpptd.neighborOrder;
        }

        FocalPointPlasticityLinkTrackerData operator=(const FocalPointPlasticityTrackerData &fpptd) {
            return FocalPointPlasticityLinkTrackerData(fpptd);
        }

        // members: link properties

        float lambdaDistance;
        float targetDistance;
        float maxDistance;
        int maxNumberOfJunctions;
        float activationEnergy;
        int neighborOrder;
        bool anchor;
        std::vector<float> anchorPoint;
        int initMCS;

        int anchorId;

    };

};
#endif