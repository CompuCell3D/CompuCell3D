#ifndef FOCALPOINTPLASTICITYLINKS_H
#define FOCALPOINTPLASTICITYLINKS_H

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/DerivedProperty.h>

#include "FocalPointPlasticityTracker.h"
#include "FocalPointPlasticityDLLSpecifier.h"

class ExpressionEvaluator;

namespace CompuCell3D {

    /**
    Written by T.J. Sego, Ph.D.
    */

    class BoundaryStrategy;

    class CellG;

    class Potts3D;

    class FocalPointPlasticityTrackerData;

    class FocalPointPlasticityLinkTrackerData;

    // Link definitions

    enum FocalPointPlasticityLinkType {
        REGULAR, INTERNAL, ANCHOR
    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityLinkBase {

    private:

        FocalPointPlasticityLinkType type;

    protected:

        Potts3D *potts;

        CellG *initiator;
        CellG *initiated;
        FocalPointPlasticityLinkTrackerData fppltd;

        ExpressionEvaluator ev;

        void initializeConstitutiveLaw(std::string _localLaw);

        bool usingLocalLaw = false;

    public:

        FocalPointPlasticityLinkBase() :
                initiator(0), initiated(0), potts(0), fppltd(FocalPointPlasticityLinkTrackerData()) {
            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getDistance> length(
                    this);
            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getTension> tension(
                    this);
            pyAttrib = 0;

        }

        ~FocalPointPlasticityLinkBase() {}

        const FocalPointPlasticityLinkType getType() { return type; }

        // Legacy support
        FocalPointPlasticityTrackerData getFPPTrackerData(CellG *_cell) {
            FocalPointPlasticityTrackerData fpptd = FocalPointPlasticityTrackerData(fppltd);
            fpptd.neighborAddress = getOtherCell(_cell);
            fpptd.isInitiator = isInitiator(_cell);
            fpptd.maxNumberOfJunctions = getMaxNumberOfJunctions();
            fpptd.activationEnergy = getActivationEnergy();
            fpptd.neighborOrder = getNeighborOrder();
            return fpptd;
        }

        // Derived properties

        // Function defining the value of derived property: length
        float getDistance();

        // Function defining the value of derived property: tension
        float getTension();

        // Length of link
        DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getDistance> length;
        // Tension in link
        DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getTension> tension;

        // General interface

        // Set the string for the constitutive law of this link
        void setConstitutiveLaw(std::string _lawString);

        // Return whether this link has a constitutive law
        const bool hasLocalLaw() { return usingLocalLaw; }

        // Pass the other cell of this link
        CellG *getOtherCell(CellG *_cell) {
            if (_cell) {
                if (initiator && _cell->id == initiator->id) return initiated;
                if (initiated && _cell->id == initiated->id) return initiator;
            } else {
                if (!initiator) return initiated;
                if (!initiated) return initiator;
            }
            throw CC3DException("Cell is not a member of this link");
        }

        CellG *getOtherCell(const CellG *_cell) { return getOtherCell(const_cast<CellG *>(_cell)); }

        // Pass whether this cell is the initiator
        bool isInitiator(CellG *_cell) {
            if (_cell) {
                if (initiator && _cell->id == initiator->id) return false;
                if (initiated && _cell->id == initiated->id) return true;
            } else {
                if (!initiator) return false;
                if (!initiated) return true;
            }
            throw CC3DException("Cell is not a member of this link");
        }

        double constitutiveLaw(float _lambda, float _length, float _targetLength);

        // Data interface

        // Get lambda distance
        const float getLambdaDistance() { return fppltd.lambdaDistance; }

        // Set lambda distance
        void setLambdaDistance(float _lambdaDistance) { fppltd.lambdaDistance = _lambdaDistance; }

        // Get target distance
        const float getTargetDistance() { return fppltd.targetDistance; }

        // Set target distance
        void setTargetDistance(float _targetDistance) { fppltd.targetDistance = _targetDistance; }

        // Get maximum distance
        const float getMaxDistance() { return fppltd.maxDistance; }

        // Set maximum distance
        void setMaxDistance(float _maxDistance) { fppltd.maxDistance = _maxDistance; }

        // Get maximum number of junctions
        const int getMaxNumberOfJunctions() { return fppltd.maxNumberOfJunctions; }

        // Set maximum number of junctions
        void setMaxNumberOfJunctions(int _maxNumberOfJunctions) { fppltd.maxNumberOfJunctions = _maxNumberOfJunctions; }

        // Get activation energy
        const float getActivationEnergy() { return fppltd.activationEnergy; }

        // Set activation energy
        void setActivationEnergy(float _activationEnergy) { fppltd.activationEnergy = _activationEnergy; }

        // Get neighbor order
        const int getNeighborOrder() { return fppltd.neighborOrder; }

        // Set neighbor order
        void setNeighborOrder(int _neighborOrder) { fppltd.neighborOrder = _neighborOrder; }

        // Get is anchor
        const bool isAnchor() { return fppltd.anchor; }

        // Get initialization step
        const int getInitMCS() { return fppltd.initMCS; }

        // Get first id
        virtual const long getId0() { return long(0); };

        // Get second id
        virtual const long getId1() { return long(0); };

        // Get first object
        CellG *getObj0() { return initiator; }

        // Get second object
        CellG *getObj1() { return initiated; }

        // Python support

        PyObject *pyAttrib;

        PyObject *getPyAttrib() {
            return pyAttrib;
        }
    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityLink : public FocalPointPlasticityLinkBase {

        FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::REGULAR;

    public:
        FocalPointPlasticityLink() {};

        FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts,
                                 FocalPointPlasticityLinkTrackerData _fppltd) {
            initiator = _initiator;
            initiated = _initiated;
            potts = _potts;
            fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
            fppltd.anchor = false;
            pyAttrib = 0;

            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getDistance> length(
                    this);
            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getTension> tension(
                    this);

            DerivedProperty<FocalPointPlasticityLink, std::vector<CellG *>, &FocalPointPlasticityLink::getCellPair> cellPair;
        }

        FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts,
                                 FocalPointPlasticityTrackerData _fpptd) :
                FocalPointPlasticityLink(_initiator, _initiated, _potts, FocalPointPlasticityLinkTrackerData(_fpptd)) {}

        FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0,
                                 float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0) :
                FocalPointPlasticityLink(_initiator, _initiated, _potts,
                                         FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance,
                                                                             _maxDistance, _initMCS)) {}

        const long getId0() { return initiator->id; }

        const long getId1() { return initiated->id; }

        std::vector<CellG *> getCellPair();

        DerivedProperty<FocalPointPlasticityLink, std::vector<CellG *>, &FocalPointPlasticityLink::getCellPair> cellPair;

    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityInternalLink : public FocalPointPlasticityLinkBase {

        FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::INTERNAL;

    public:
        FocalPointPlasticityInternalLink() {};

        FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts,
                                         FocalPointPlasticityLinkTrackerData _fppltd) {
            initiator = _initiator;
            initiated = _initiated;
            potts = _potts;
            fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
            fppltd.anchor = false;
            pyAttrib = 0;

            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getDistance> length(
                    this);
            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getTension> tension(
                    this);

            DerivedProperty<FocalPointPlasticityInternalLink, std::vector<CellG *>, &FocalPointPlasticityInternalLink::getCellPair> cellPair(
                    this);
        }

        FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts,
                                         FocalPointPlasticityTrackerData _fpptd) :
                FocalPointPlasticityInternalLink(_initiator, _initiated, _potts,
                                                 FocalPointPlasticityLinkTrackerData(_fpptd)) {}

        FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts,
                                         float _lambdaDistance = 0.0, float _targetDistance = 0.0,
                                         float _maxDistance = 100000.0, int _initMCS = 0) :
                FocalPointPlasticityInternalLink(_initiator, _initiated, _potts,
                                                 FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance,
                                                                                     _maxDistance, _initMCS)) {}

        const long getId0() { return initiator->id; }

        const long getId1() { return initiated->id; }

        std::vector<CellG *> getCellPair();

        DerivedProperty<FocalPointPlasticityInternalLink, std::vector<CellG *>, &FocalPointPlasticityInternalLink::getCellPair> cellPair;

    };

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityAnchor : public FocalPointPlasticityLinkBase {

        FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::ANCHOR;

    public:
        FocalPointPlasticityAnchor() {}

        FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd) {
            initiator = _cell;
            initiated = (CellG *) (0);
            potts = _potts;
            fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
            fppltd.anchor = true;
            fppltd.anchorPoint = _fppltd.anchorPoint;
            pyAttrib = 0;

            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getDistance> length(
                    this);
            DerivedProperty<FocalPointPlasticityLinkBase, float, &FocalPointPlasticityLinkBase::getTension> tension(
                    this);

            DerivedProperty < FocalPointPlasticityLinkBase, CellG *, &FocalPointPlasticityLinkBase::getObj0 >
                                                                     cell(this);
        }

        FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
                FocalPointPlasticityAnchor(_cell, _potts, FocalPointPlasticityLinkTrackerData(_fpptd)) {}

        FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, float _lambdaDistance = 0.0,
                                   float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0,
                                   std::vector<float> _anchorPoint = std::vector<float>(3, 0.0)) :
                FocalPointPlasticityAnchor(_cell, _potts,
                                           FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance,
                                                                               _maxDistance, _initMCS)) {}

        const long getId0() { return initiator->id; }

        const long getId1() { return fppltd.anchorId; }

        // Get anchor point
        std::vector<float> getAnchorPoint() { return fppltd.anchorPoint; }

        // Set anchor point
        void setAnchorPoint(std::vector<float> _anchorPoint) { fppltd.anchorPoint = _anchorPoint; }

        // Get anchor id
        const int getAnchorId() { return fppltd.anchorId; }

        DerivedProperty<FocalPointPlasticityLinkBase, CellG *, &FocalPointPlasticityLinkBase::getObj0> cell;

    };

}

#endif