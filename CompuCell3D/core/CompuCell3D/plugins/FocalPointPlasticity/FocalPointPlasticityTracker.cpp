#include "FocalPointPlasticityTracker.h"

using namespace CompuCell3D;

FocalPointPlasticityTrackerData::FocalPointPlasticityTrackerData(const FocalPointPlasticityLinkTrackerData &fppltd) {
    lambdaDistance = fppltd.lambdaDistance;
    targetDistance = fppltd.targetDistance;
    maxDistance = fppltd.maxDistance;
    activationEnergy = fppltd.activationEnergy;
    maxNumberOfJunctions = fppltd.maxNumberOfJunctions;
    neighborOrder = fppltd.neighborOrder;
    anchor = fppltd.anchor;
    anchorId = fppltd.anchorId;
    anchorPoint = fppltd.anchorPoint;
    initMCS = fppltd.initMCS;
}