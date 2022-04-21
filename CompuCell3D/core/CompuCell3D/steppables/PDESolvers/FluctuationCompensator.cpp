
#include "FluctuationCompensator.h"

using namespace CompuCell3D;

FluctuationCompensator::FluctuationCompensator(Simulator *_sim) {
    sim = _sim;
    potts = _sim->getPotts();
    automaton = potts->getAutomaton();
    cellInventory = &potts->getCellInventory();
    pUtils = sim->getParallelUtils();

    fieldDim = potts->getCellFieldG()->getDim();

    diffusibleFields.clear();
    diffusibleFieldNames.clear();
    numFields = 0;

    bool pluginAlreadyRegisteredFlag;
    pixelTrackerPlugin = (PixelTrackerPlugin *) Simulator::pluginManager.get("PixelTracker",
                                                                             &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag) {
        CC3DXMLElement *pixelTrackerXML = sim->getCC3DModuleData("Plugin", "PixelTracker");
        pixelTrackerPlugin->init(sim, pixelTrackerXML);
    }
    pixelTrackerPlugin->enableMediumTracker();

    potts->registerCellGChangeWatcher(this);

    needsInitialized = true;

}

FluctuationCompensator::~FluctuationCompensator() {
    for (cellCompensatorDataItr = cellCompensatorData.begin();
         cellCompensatorDataItr != cellCompensatorData.end(); ++cellCompensatorDataItr) {
        delete cellCompensatorDataItr->second;
        cellCompensatorDataItr->second = 0;
    }
}

////////////////////////////////////////////////////// Cell field watcher interface //////////////////////////////////////////////////////

void FluctuationCompensator::field3DChange(const Point3D &pt, const Point3D &addPt, CellG *newCell, CellG *oldCell) {

    // Get concentrations at copy-associated sites
    std::vector<float> concentrationVecOld = getConcentrationVec(pt);
    std::vector<float> concentrationVecNew = getConcentrationVec(addPt);

    if (oldCell) {

        // Get cell data
        FluctuationCompensatorCellData &fccdOld = *getFluctuationCompensatorCellData(oldCell);

        // Update correction factors
        for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
            fccdOld.concentrationVecCopies[fieldIndex] -= concentrationVecOld[fieldIndex];

    } else {
        // Update correction factors
        for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
            concentrationVecCopiesMediumTheads[pUtils->getCurrentWorkNodeNumber()][fieldIndex] -= concentrationVecOld[fieldIndex];
    }

    if (newCell) {

        // Get cell data
        FluctuationCompensatorCellData &fccdNew = *getFluctuationCompensatorCellData(newCell);

        // Update correction factors
        for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
            fccdNew.concentrationVecCopies[fieldIndex] += concentrationVecNew[fieldIndex];

    } else {
        // Update correction factors
        for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
            concentrationVecCopiesMediumTheads[pUtils->getCurrentWorkNodeNumber()][fieldIndex] += concentrationVecNew[fieldIndex];

    }

    setConcentrationVec(pt, concentrationVecNew);

}

//////////////////////////////////////////////////////////// Solver interface ////////////////////////////////////////////////////////////

void FluctuationCompensator::loadFieldName(std::string _fieldName) {
    diffusibleFieldNames.push_back(_fieldName);
    ++numFields;
}

void FluctuationCompensator::loadFields() {
    diffusibleFields.assign(numFields, 0);
    std::map < std::string, Field3D<float> * > concentrationFieldNameMap = sim->getConcentrationFieldNameMap();
    for (int fieldIdx = 0; fieldIdx < diffusibleFieldNames.size(); ++fieldIdx)
        diffusibleFields[fieldIdx] = (Field3DImpl<float> *) concentrationFieldNameMap[diffusibleFieldNames[fieldIdx]];
    resetCorrections();
}

void FluctuationCompensator::applyCorrections() {

    // Since this is called by a solver before integration, it's a thread-safe place to do global initializations and updates

    // On-the-fly initiailzation

    if (needsInitialized) {
        resetCorrections();

        needsInitialized = false;
    }

    // Apply corrections

    // Gather worker data

    for (unsigned int i = 0; i < concentrationVecCopiesMediumTheads.size(); ++i)
        for (unsigned int j = 0; j < numFields; ++j)
            concentrationVecCopiesMedium[j] += concentrationVecCopiesMediumTheads[i][j];

    // Get worker medium sets

    std::vector <std::set<PixelTrackerData>> pixelWorkerSets = pixelTrackerPlugin->getPixelWorkerSets();
    unsigned int numWorkers = pixelWorkerSets.size();

    // Pre-calculate corrections for the medium and share

    std::vector<float> correctionFactorsTmp(numFields, 0.0);
    for (unsigned int i = 0; i < numFields; ++i) {
        float den = concentrationVecTotalMedium[i] + concentrationVecCopiesMedium[i];
        if (den != 0) correctionFactorsTmp[i] = concentrationVecTotalMedium[i] / den;
    }
    std::vector <std::vector<float>> correctionFactorsMedium(numWorkers, correctionFactorsTmp);

    // Build vector of cell for sharing

    unsigned int numCells = cellInventory->getSize();
    std::vector < std::tuple < FluctuationCompensatorCellData * , std::set < PixelTrackerData > > >
                                                                  cellDataVec(numCells);
    unsigned int cellIdx = 0;
    for (CellInventory::cellInventoryIterator itr = cellInventory->cellInventoryBegin();
         itr != cellInventory->cellInventoryEnd(); ++itr) {
        CellG *cell = cellInventory->getCell(itr);
        FluctuationCompensatorCellData *fccd = cellCompensatorData[cell];
        std::set <PixelTrackerData> pixelSet = pixelTrackerPlugin->getPixelTrackerAccessorPtr()->get(
                cell->extraAttribPtr)->pixelSet;
        cellDataVec[cellIdx] = make_tuple(fccd, pixelSet);
        ++cellIdx;
    }

    // Do parallel loop over worker medium subdomains

#pragma omp parallel
    {
#pragma omp for nowait
        for (int workerNum = 0; workerNum < numWorkers; ++workerNum) {
            std::set <PixelTrackerData> pixelSetLocal = pixelWorkerSets[workerNum];
            std::vector<float> correctionFactorsLocal = correctionFactorsMedium[workerNum];

            for (auto ptd: pixelSetLocal)
                for (int fieldIdx = 0; fieldIdx < diffusibleFields.size(); ++fieldIdx)
                    diffusibleFields[fieldIdx]->set(ptd.pixel, (correctionFactorsLocal[fieldIdx] *
                                                                diffusibleFields[fieldIdx]->get(ptd.pixel)));
        }

        // Do loop over cell subdomains
#pragma omp for
        for (int cellIdx = 0; cellIdx < cellDataVec.size(); ++cellIdx) {
            std::tuple < FluctuationCompensatorCellData * ,
                    std::set < PixelTrackerData > > cellData = cellDataVec[cellIdx];
            FluctuationCompensatorCellData *fccd = std::get<0>(cellData);
            std::set <PixelTrackerData> pixelSetLocal = std::get<1>(cellData);
            std::vector<float> concentrationVecCopiesLocal = fccd->concentrationVecCopies;
            std::vector<float> concentrationVecTotalsLocal = fccd->concentrationVecTotals;

            unsigned int numFieldsLocal = concentrationVecCopiesLocal.size();

            std::vector<float> correctionFactorsLocal(numFieldsLocal, 0.0);
            for (unsigned int i = 0; i < numFieldsLocal; ++i) {
                float den = concentrationVecTotalsLocal[i] + concentrationVecCopiesLocal[i];
                if (den != 0) correctionFactorsLocal[i] = concentrationVecTotalsLocal[i] / den;
            }

            for (auto ptd: pixelSetLocal) {
                for (int fieldIdx = 0; fieldIdx < diffusibleFields.size(); ++fieldIdx)
                    diffusibleFields[fieldIdx]->set(ptd.pixel, (correctionFactorsLocal[fieldIdx] *
                                                                diffusibleFields[fieldIdx]->get(ptd.pixel)));
            }

        }

    }

}

void FluctuationCompensator::resetCorrections() {
    resetCellConcentrations();
    resetMediumConcentration();
}

void FluctuationCompensator::resetCellConcentrations() {

    for (CellInventory::cellInventoryIterator cItr = cellInventory->cellInventoryBegin();
         cItr != cellInventory->cellInventoryEnd(); ++cItr) {
        CellG *cell = cItr->second;

        FluctuationCompensatorCellData &cData = *getFluctuationCompensatorCellData(cell, false);
        cData.concentrationVecTotals = totalCellConcentration(cell);
        cData.concentrationVecCopies = std::vector<float>(numFields, 0.0);
    }
}

void FluctuationCompensator::resetMediumConcentration() {
    concentrationVecTotalMedium = totalMediumConcentration();
    concentrationVecCopiesMedium = std::vector<float>(numFields, 0.0);
    concentrationVecCopiesMediumTheads.assign(pUtils->getMaxNumberOfWorkNodesPotts(), concentrationVecCopiesMedium);
}

void FluctuationCompensator::updateTotalConcentrations() {
    updateTotalCellConcentrations();
    updateTotalMediumConcentration();
}

void FluctuationCompensator::updateTotalCellConcentrations() {

    for (CellInventory::cellInventoryIterator cItr = cellInventory->cellInventoryBegin();
         cItr != cellInventory->cellInventoryEnd(); ++cItr) {
        CellG *cell = cItr->second;

        FluctuationCompensatorCellData &cData = *getFluctuationCompensatorCellData(cell, false);
        cData.concentrationVecTotals = totalCellConcentration(cell);
    }
}

void FluctuationCompensator::updateTotalMediumConcentration() {
    concentrationVecTotalMedium = totalMediumConcentration();
}

std::vector<float> FluctuationCompensator::totalCellConcentration(const CellG *_cell) {
    return totalPixelSetConcentration(getCellPixelVec(_cell));
}

std::vector<float> FluctuationCompensator::totalMediumConcentration() {
    return totalPixelSetConcentration(getMediumPixelVec());
}

std::vector <Point3D> FluctuationCompensator::getCellPixelVec(const CellG *_cell) {
    std::set <PixelTrackerData> &pixelSet = pixelTrackerPlugin->getPixelTrackerAccessorPtr()->get(
            _cell->extraAttribPtr)->pixelSet;
    std::vector <Point3D> pixelVec = std::vector<Point3D>(pixelSet.size());
    unsigned int ptdIndex = 0;
    for (set<PixelTrackerData>::iterator sitr = pixelSet.begin(); sitr != pixelSet.end(); ++sitr) {
        pixelVec[ptdIndex] = sitr->pixel;
        ++ptdIndex;
    }
    return pixelVec;
}

std::vector <Point3D> FluctuationCompensator::getMediumPixelVec() {
    const std::set <PixelTrackerData> &pixelSet = pixelTrackerPlugin->getMediumPixelSet();
    std::vector <Point3D> pixelVec = std::vector<Point3D>(pixelSet.size());
    unsigned int ptdIndex = 0;
    for (set<PixelTrackerData>::iterator sitr = pixelSet.begin(); sitr != pixelSet.end(); ++sitr) {
        pixelVec[ptdIndex] = sitr->pixel;
        ++ptdIndex;
    }
    return pixelVec;
}

std::vector<float> FluctuationCompensator::totalPixelSetConcentration(std::vector <Point3D> _pixelVec) {
    std::vector<float> sumVec = std::vector<float>(numFields, 0.0);
    for (unsigned int pixelIndex = 0; pixelIndex < _pixelVec.size(); ++pixelIndex) {
        for (unsigned int vecIndex = 0; vecIndex < numFields; ++vecIndex) {
            sumVec[vecIndex] += diffusibleFields[vecIndex]->get(_pixelVec[pixelIndex]);
        }
    }
    return sumVec;
}

FluctuationCompensatorCellData *
FluctuationCompensator::getFluctuationCompensatorCellData(CellG *_cell, bool _fullInit) {
    FluctuationCompensatorCellData *cData;
    cellCompensatorDataItr = cellCompensatorData.find(_cell);
    if (cellCompensatorDataItr == cellCompensatorData.end()) {
        // On-the-fly initializations
        if (_cell) {
            cData = new FluctuationCompensatorCellData(numFields);
            if (_fullInit) cData->concentrationVecTotals = totalCellConcentration(_cell);
            cellCompensatorData.insert(make_pair(_cell, cData));
        }
    } else cData = cellCompensatorDataItr->second;
    return cData;
}

std::vector<float> FluctuationCompensator::getConcentrationVec(const Point3D &_pt) {
    std::vector<float> _res = std::vector<float>(numFields, 0.0);
    for (int _idx = 0; _idx < diffusibleFields.size(); ++_idx) _res[_idx] = diffusibleFields[_idx]->get(_pt);
    return _res;
}

void FluctuationCompensator::setConcentrationVec(const Point3D &_pt, std::vector<float> _vec) {
    for (int _idx = 0; _idx < diffusibleFields.size(); ++_idx) diffusibleFields[_idx]->set(_pt, _vec[_idx]);
}
