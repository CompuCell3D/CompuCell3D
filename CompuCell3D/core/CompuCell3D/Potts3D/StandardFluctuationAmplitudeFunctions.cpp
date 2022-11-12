#include "StandardFluctuationAmplitudeFunctions.h"
#include "Potts3D.h"
#include "Cell.h"

#undef max
#undef min

#include <algorithm>
#include <limits>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;
using namespace std;

MinFluctuationAmplitudeFunction::MinFluctuationAmplitudeFunction(const Potts3D *_potts) :
        FluctuationAmplitudeFunction(_potts) {
}

double MinFluctuationAmplitudeFunction::fluctuationAmplitude(const CellG *newCell, const CellG *oldCell) {
    std::vector<double> fluctAmplVec(2, 0.0);
    double fluctAmpl = potts->getTemperature();

    //first we check if users defined fluctuation by cell type

    if (potts->hasCellTypeMotility()) {
        unsigned int newCellTypeId = (newCell ? (unsigned int) newCell->type : 0);
        unsigned int oldCellTypeId = (oldCell ? (unsigned int) oldCell->type : 0);
        if (newCellTypeId && oldCellTypeId) {

            fluctAmplVec[0] = (newCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(newCellTypeId)
                                                        : newCell->fluctAmpl);
            fluctAmplVec[1] = (oldCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(oldCellTypeId)
                                                        : oldCell->fluctAmpl);

        } else if (newCellTypeId) {

            fluctAmplVec[0] = (newCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(newCellTypeId)
                                                        : newCell->fluctAmpl);
            fluctAmplVec[1] = (std::numeric_limits<double>::max)();

        } else if (oldCellTypeId) {

            fluctAmplVec[0] = (std::numeric_limits<double>::max)();
            fluctAmplVec[1] = (oldCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(oldCellTypeId)
                                                        : oldCell->fluctAmpl);

        }

        return *min_element(fluctAmplVec.begin(), fluctAmplVec.end());

    }

    //check if user modified fluctAmpl attribute
    bool globalFluctuationAmplitudeFlag = true;

    if (oldCell && oldCell->fluctAmpl >= 0.0) {
        globalFluctuationAmplitudeFlag = false;
    }

    if (newCell && newCell->fluctAmpl >= 0.0) {
        globalFluctuationAmplitudeFlag = globalFluctuationAmplitudeFlag && false;
    }


    if (globalFluctuationAmplitudeFlag) { //if no
        return potts->getTemperature();
    } else {//if any of fluctAmpl attribute has been modified

        if (newCell) {
            if (newCell->fluctAmpl >= 0.0)
                fluctAmplVec[0] = newCell->fluctAmpl;
            else
                fluctAmplVec[0] = potts->getTemperature();
        } else {

            fluctAmplVec[0] = (std::numeric_limits<double>::max)();
        }

        if (oldCell) {
            if (oldCell->fluctAmpl >= 0.0)
                fluctAmplVec[1] = oldCell->fluctAmpl;
            else
                fluctAmplVec[1] = potts->getTemperature();
        } else {

            fluctAmplVec[1] = (std::numeric_limits<double>::max)();
        }


        return *min_element(fluctAmplVec.begin(), fluctAmplVec.end());
    }

    CC3D_Log(LOG_DEBUG) << "MIN FLUCT AMPL: RETURNING TEMP=" << potts->getTemperature();
    return potts->getTemperature(); //should never get here
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MaxFluctuationAmplitudeFunction::MaxFluctuationAmplitudeFunction(const Potts3D *_potts) :
        FluctuationAmplitudeFunction(_potts) {
}

double MaxFluctuationAmplitudeFunction::fluctuationAmplitude(const CellG *newCell, const CellG *oldCell) {
    std::vector<double> fluctAmplVec(2, 0.0);
    double fluctAmpl = potts->getTemperature();

    //first we check if users defined fluctuation by cell type

    if (potts->hasCellTypeMotility()) {
        unsigned int newCellTypeId = (newCell ? (unsigned int) newCell->type : 0);
        unsigned int oldCellTypeId = (oldCell ? (unsigned int) oldCell->type : 0);
        if (newCellTypeId && oldCellTypeId) {

            fluctAmplVec[0] = (newCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(newCellTypeId)
                                                        : newCell->fluctAmpl);
            fluctAmplVec[1] = (oldCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(oldCellTypeId)
                                                        : oldCell->fluctAmpl);

        } else if (newCellTypeId) {

            fluctAmplVec[0] = (newCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(newCellTypeId)
                                                        : newCell->fluctAmpl);
            fluctAmplVec[1] = (std::numeric_limits<double>::min)();

        } else if (oldCellTypeId) {

            fluctAmplVec[0] = (std::numeric_limits<double>::min)();
            fluctAmplVec[1] = (oldCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(oldCellTypeId)
                                                        : oldCell->fluctAmpl);

        }

        return *max_element(fluctAmplVec.begin(), fluctAmplVec.end());

    }

    //check if user modified fluctAmpl attribute
    bool globalFluctuationAmplitudeFlag = true;

    if (oldCell && oldCell->fluctAmpl >= 0.0) {
        globalFluctuationAmplitudeFlag = false;
    }

    if (newCell && newCell->fluctAmpl >= 0.0) {
        globalFluctuationAmplitudeFlag = globalFluctuationAmplitudeFlag && false;
    }


    if (globalFluctuationAmplitudeFlag) { //if no
        return potts->getTemperature();
    } else {//if any of fluctAmpl attribute has been modified

        if (newCell) {
            if (newCell->fluctAmpl >= 0.0)
                fluctAmplVec[0] = newCell->fluctAmpl;
            else
                fluctAmplVec[0] = potts->getTemperature();
        } else {

            fluctAmplVec[0] = (std::numeric_limits<double>::min)();
        }

        if (oldCell) {
            if (oldCell->fluctAmpl >= 0.0)
                fluctAmplVec[1] = oldCell->fluctAmpl;
            else
                fluctAmplVec[1] = potts->getTemperature();
        } else {

            fluctAmplVec[1] = (std::numeric_limits<double>::min)();
        }


        return *max_element(fluctAmplVec.begin(), fluctAmplVec.end());
    }

	CC3D_Log(LOG_DEBUG) << "MAX FLUCT AMPL : RETURNING TEMP="<<potts->getTemperature();
    return potts->getTemperature(); //should never get here
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ArithmeticAverageFluctuationAmplitudeFunction::ArithmeticAverageFluctuationAmplitudeFunction(const Potts3D *_potts) :
        FluctuationAmplitudeFunction(_potts) {
}

double ArithmeticAverageFluctuationAmplitudeFunction::fluctuationAmplitude(const CellG *newCell, const CellG *oldCell) {
    std::vector<double> fluctAmplVec(2, 0.0);
    double fluctAmpl = potts->getTemperature();

    //first we check if users defined fluctuation by cell type

    if (potts->hasCellTypeMotility()) {
        unsigned int newCellTypeId = (newCell ? (unsigned int) newCell->type : 0);
        unsigned int oldCellTypeId = (oldCell ? (unsigned int) oldCell->type : 0);
        if (newCellTypeId && oldCellTypeId) {

            fluctAmplVec[0] = (newCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(newCellTypeId)
                                                        : newCell->fluctAmpl);
            fluctAmplVec[1] = (oldCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(oldCellTypeId)
                                                        : oldCell->fluctAmpl);

        } else if (newCellTypeId) {

            fluctAmplVec[0] = (newCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(newCellTypeId)
                                                        : newCell->fluctAmpl);
            fluctAmplVec[1] = fluctAmplVec[0];

        } else if (oldCellTypeId) {

            fluctAmplVec[1] = (oldCell->fluctAmpl < 0.0 ? potts->getCellTypeMotility(oldCellTypeId)
                                                        : oldCell->fluctAmpl);
            fluctAmplVec[0] = fluctAmplVec[1];


        }

        return (fluctAmplVec[0] + fluctAmplVec[1]) / 2.0;

    }

    //check if user modified fluctAmpl attribute
    bool globalFluctuationAmplitudeFlag = true;

    if (oldCell && oldCell->fluctAmpl >= 0.0) {
        globalFluctuationAmplitudeFlag = false;
    }

    if (newCell && newCell->fluctAmpl >= 0.0) {
        globalFluctuationAmplitudeFlag = globalFluctuationAmplitudeFlag && false;
    }


    if (globalFluctuationAmplitudeFlag) { //if no
        return potts->getTemperature();
    } else {//if any of fluctAmpl attribute has been modified

        if (newCell) {
            if (newCell->fluctAmpl >= 0.0)
                fluctAmplVec[0] = newCell->fluctAmpl;
            else
                fluctAmplVec[0] = potts->getTemperature();
        } else {

            fluctAmplVec[0] = -1.0;
        }

        if (oldCell) {
            if (oldCell->fluctAmpl >= 0.0)
                fluctAmplVec[1] = oldCell->fluctAmpl;
            else
                fluctAmplVec[1] = potts->getTemperature();
        } else {

            fluctAmplVec[1] = -1.0;
        }
        // it main potts algorith
        if (fluctAmplVec[0] < 0.0) {
            fluctAmplVec[0] = fluctAmplVec[1];
        } else if (fluctAmplVec[1] < 0.0) {
            fluctAmplVec[1] = fluctAmplVec[0];
        }


        return (fluctAmplVec[0] + fluctAmplVec[1]) / 2.0;
    }

    CC3D_Log(LOG_DEBUG) << "ARITHMETIC AVERAGE FLUCT AMPL : RETURNING TEMP=" << potts->getTemperature();
    return potts->getTemperature(); //should never get here
}