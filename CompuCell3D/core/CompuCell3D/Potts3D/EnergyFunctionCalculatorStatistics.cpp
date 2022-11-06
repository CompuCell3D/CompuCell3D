#include "EnergyFunctionCalculatorStatistics.h"
#include "EnergyFunction.h"
#include <iterator>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <CompuCell3D/Simulator.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <sstream>
#include <CompuCell3D/PottsParseData.h>
#include <XMLUtils/CC3DXMLElement.h>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;
using namespace std;

EnergyFunctionCalculatorStatistics::EnergyFunctionCalculatorStatistics() : EnergyFunctionCalculator() {

    NTot = 0;
    NAcc = 0;
    NRej = 0;
    lastFlipAttempt = -1;
    out = 0;
    outAccSpinFlip = 0;
    outRejSpinFlip = 0;
    outTotSpinFlip = 0;

    wroteHeader = false;
    outputEverySpinFlip = false;
    gatherResultsSpinFlip = false;
    outputAcceptedSpinFlip = false;
    outputRejectedSpinFlip = false;
    outputTotalSpinFlip = false;

    gatherResultsFilesPrepared = false;

    mcs = 0;
    fieldWidth = 30;
    analysisFrequency = 1;
    singleSpinFrequency = 1;
}

EnergyFunctionCalculatorStatistics::~EnergyFunctionCalculatorStatistics() {
    if (out) {
        out->close();
        delete out;
        out = 0;

    }

    if (outAccSpinFlip) {
        outAccSpinFlip->close();
        delete outAccSpinFlip;
        outAccSpinFlip = 0;
    }

    if (outRejSpinFlip) {
        outRejSpinFlip->close();
        delete outRejSpinFlip;
        outRejSpinFlip = 0;
    }

    if (outTotSpinFlip) {
        outTotSpinFlip->close();
        delete outTotSpinFlip;
        outTotSpinFlip = 0;
    }

}


double EnergyFunctionCalculatorStatistics::changeEnergy(Point3D &pt, const CellG *newCell, const CellG *oldCell,
                                                        const unsigned int _flipAttempt) {

    ParallelUtilsOpenMP *pUtils = sim->getParallelUtils();

    if (pUtils->getNumberOfWorkNodesPotts() == 1) {
        if (lastFlipAttempt < 0) {
            initialize();
        }


        if (_flipAttempt < lastFlipAttempt) {
            current_mcs_pos = totEnergyDataList.size();
            ++mcs;

            totEnergyDataListCurrent.clear();
            accNotAccListCurrent.clear();

            if (!(mcs % analysisFrequency)) {
                outputResults();
            }
            if (!gatherResultsSpinFlip && outputEverySpinFlip && !(mcs % singleSpinFrequency)) {
                outputResultsSingleSpinFlip();
            }

            if (gatherResultsSpinFlip && outputEverySpinFlip && !(mcs % singleSpinFrequency)) {
                if (!gatherResultsFilesPrepared) {
                    prepareGatherResultsFiles();
                }
                outputResultsSingleSpinFlipGatherResults();
            }

            prepareNextStep();
        }


        lastFlipAttempt = _flipAttempt;
    }
    double change = 0;

    if (pUtils->getNumberOfWorkNodesPotts() == 1) {

        for (unsigned int i = 0; i < energyFunctions.size(); i++) {
            lastEnergyVec[i] = energyFunctions[i]->changeEnergy(pt, newCell, oldCell);
            change += lastEnergyVec[i];

        }
        mcs_list.push_back(mcs);
        pixel_copy_attempt_points_list.push_back(pt);
        totEnergyDataList.push_back(
                lastEnergyVec); //inserting lastEnergyVec into  totEnergyVecVec - for later stdDev calculations
        totEnergyDataListCurrent.push_back(lastEnergyVec);

    } else {
        for (unsigned int i = 0; i < energyFunctions.size(); i++) {
            change += energyFunctions[i]->changeEnergy(pt, newCell, oldCell);
            CC3D_Log(LOG_DEBUG) << "CHANGE FROM ACCEPTANCE FUNCTION"<<change<<" FCNNAME="<<energyFunctionsNameVec[i];
        }

    }
    return change;

}

void EnergyFunctionCalculatorStatistics::range(int *rangevec, int n) {
    int i;

    for (i = 0; i < n; i++)
        rangevec[i] = i;
}

long EnergyFunctionCalculatorStatistics::get_number_energy_fcn_calculations() {
    return accNotAccList.size();
}

void EnergyFunctionCalculatorStatistics::get_current_mcs_accepted_mask_npy_array(int *intvec, int n) {

    std::list<bool>::iterator accepted_flip_mask_list_litr = accNotAccList.begin();

    //advance(accepted_flip_mask_list_litr, current_mcs_pos); //advancing itr to current position
    for (long i = 0; i < n; ++i) {
        intvec[i] = *accepted_flip_mask_list_litr;
        ++accepted_flip_mask_list_litr;
    }

}

void EnergyFunctionCalculatorStatistics::get_current_mcs_prob_npy_array(double *doublevec, int n) {

    std::list<double>::iterator acceptance_probability_list_litr = acceptance_probability_list.begin();

    //advance(acceptance_probability_list_litr, current_mcs_pos); //advancing itr to current position
    for (long i = 0; i < n; ++i) {
        doublevec[i] = *acceptance_probability_list_litr;
        ++acceptance_probability_list_litr;
    }


}

void EnergyFunctionCalculatorStatistics::get_current_mcs_flip_attempt_points_npy_array(short *shortvec, int n) {
    CC3D_Log(LOG_DEBUG) << "get_current_mcs_flip_attempt_points_npy_array n=" << n;
	std::list<Point3D>::iterator pixel_copy_litr = pixel_copy_attempt_points_list.begin();
	for (long i = 0; i < n;  i+=3) {
		Point3D & pt = *pixel_copy_litr;
		shortvec[i] = pt.x;
		shortvec[i+1] = pt.y;
		shortvec[i + 2] = pt.z;
		++pixel_copy_litr;
		
	}


}


void EnergyFunctionCalculatorStatistics::set_acceptance_probability(double _prob) {
    acceptance_probability_list.push_back(_prob);
}

//this function will inccrement appropriate energy vectors based on whether flip was accepted or not
void EnergyFunctionCalculatorStatistics::setLastFlipAccepted(bool _accept) {
    lastFlipAccepted = _accept;

    if (lastFlipAccepted) {
        accNotAccList.push_back(true);
        accNotAccListCurrent.push_back(true);

    } else {
        accNotAccList.push_back(false);
        accNotAccListCurrent.push_back(false);

    }


}

void EnergyFunctionCalculatorStatistics::writeHeader() {

    (*out) << setw(fieldWidth) << "1.STEP" << setw(fieldWidth) << "2.NAcc" << setw(fieldWidth) << "3.NRej"
           << setw(fieldWidth) << "4.NTot";
    int n = 5;
    for (int i = 0; i < energyFunctions.size(); ++i) {
        ostringstream str1;
        ostringstream str2;
        str1 << n << ".Acc_" << energyFunctionsNameVec[i] << "_AVG";
        str2 << n + 1 << ". +/-";
        (*out) << setw(fieldWidth) << str1.str() << setw(fieldWidth) << str2.str();
        n += 2;
    }

    for (int i = 0; i < energyFunctions.size(); ++i) {
        ostringstream str1;
        ostringstream str2;
        str1 << n << ".Rej_" << energyFunctionsNameVec[i] << "_AVG";
        str2 << n + 1 << ". +/-";
        (*out) << setw(fieldWidth) << str1.str() << setw(fieldWidth) << str2.str();
        n += 2;
    }

    for (int i = 0; i < energyFunctions.size(); ++i) {
        ostringstream str1;
        ostringstream str2;
        str1 << n << ".Tot_" << energyFunctionsNameVec[i] << "_AVG";
        str2 << n + 1 << ". +/-";
        (*out) << setw(fieldWidth) << str1.str() << setw(fieldWidth) << str2.str();
        n += 2;
    }


    (*out) << endl;
}


void EnergyFunctionCalculatorStatistics::initialize() {
    if (!wroteHeader && out) {
        writeHeader();
        wroteHeader = true;
    }
    NAcc = 0;
    NRej = 0;
    NTot = 0;

    int energyFunctionsVecSize = energyFunctions.size();

    lastEnergyVec.assign(energyFunctionsVecSize, 0.0);
    stdDevEnergyVectorTot.assign(energyFunctionsVecSize, 0.0);
    stdDevEnergyVectorAcc.assign(energyFunctionsVecSize, 0.0);
    stdDevEnergyVectorRej.assign(energyFunctionsVecSize, 0.0);


    //allocating vectors needed for std_dev calculations


    totEnergyDataList.clear();
    totEnergyDataListCurrent.clear();
    accNotAccList.clear();
    acceptance_probability_list.clear();
    pixel_copy_attempt_points_list.clear();
    mcs_list.clear();


}

void EnergyFunctionCalculatorStatistics::prepareNextStep() {

    initialize();

}


void EnergyFunctionCalculatorStatistics::calculateStatData() {
    unsigned int energyFunctionCount = energyFunctions.size();

    avgEnergyVectorTot.assign(energyFunctionCount, 0.0);
    stdDevEnergyVectorTot.assign(energyFunctionCount, 0.0);

    avgEnergyVectorAcc.assign(energyFunctionCount, 0.0);
    stdDevEnergyVectorAcc.assign(energyFunctionCount, 0.0);

    avgEnergyVectorRej.assign(energyFunctionCount, 0.0);
    stdDevEnergyVectorRej.assign(energyFunctionCount, 0.0);




    //calculating averages
    std::list < std::vector < double > > ::iterator
    lVecItr;
    std::list<bool>::iterator lItr;

    NTot = 0;
    NAcc = 0;
    NRej = 0;

    //calculating averages
    lItr = accNotAccList.begin();
    for (lVecItr = totEnergyDataList.begin(); lVecItr != totEnergyDataList.end(); ++lVecItr) {
        ++NTot;
        vector<double> &energyData = *lVecItr;
        for (int i = 0; i < energyFunctionCount; ++i) {
            avgEnergyVectorTot[i] += energyData[i];
        }
        if (*lItr) {//accepted flip
            ++NAcc;
            for (int i = 0; i < energyFunctionCount; ++i) {
                avgEnergyVectorAcc[i] += energyData[i];
            }
        } else {//rejected flip
            ++NRej;
            for (int i = 0; i < energyFunctionCount; ++i) {
                avgEnergyVectorRej[i] += energyData[i];
            }
        }
        ++lItr;
    }

    for (int i = 0; i < energyFunctions.size(); ++i) {
        if (NTot)
            avgEnergyVectorTot[i] /= NTot;
        if (NAcc)
            avgEnergyVectorAcc[i] /= NAcc;
        if (NRej)
            avgEnergyVectorRej[i] /= NRej;
    }

    //calculating stdDev
    stdDevEnergyVectorTot.assign(energyFunctionCount, 0.);
    stdDevEnergyVectorAcc.assign(energyFunctionCount, 0.);
    stdDevEnergyVectorRej.assign(energyFunctionCount, 0.);

    lItr = accNotAccList.begin();
    for (lVecItr = totEnergyDataList.begin(); lVecItr != totEnergyDataList.end(); ++lVecItr) {
        vector<double> &energyData = *lVecItr;
        for (int i = 0; i < energyFunctionCount; ++i) {
            stdDevEnergyVectorTot[i] +=
                    (energyData[i] - avgEnergyVectorTot[i]) * (energyData[i] - avgEnergyVectorTot[i]);
        }
        if (*lItr) {//accepted flip
            for (int i = 0; i < energyFunctionCount; ++i) {
                stdDevEnergyVectorAcc[i] +=
                        (energyData[i] - avgEnergyVectorAcc[i]) * (energyData[i] - avgEnergyVectorAcc[i]);
            }
        } else {//rejected flip
            for (int i = 0; i < energyFunctionCount; ++i) {
                stdDevEnergyVectorRej[i] +=
                        (energyData[i] - avgEnergyVectorRej[i]) * (energyData[i] - avgEnergyVectorRej[i]);
            }
        }
        ++lItr;
    }



    // "Normalizing" stdDev
    for (int j = 0; j < energyFunctions.size(); ++j) {
        if (NTot)
            stdDevEnergyVectorTot[j] = sqrt(stdDevEnergyVectorTot[j]) / NTot;
        if (NAcc)
            stdDevEnergyVectorAcc[j] = sqrt(stdDevEnergyVectorAcc[j]) / NAcc;
        if (NRej)
            stdDevEnergyVectorRej[j] = sqrt(stdDevEnergyVectorRej[j]) / NRej;

    }


}

void EnergyFunctionCalculatorStatistics::writeHeaderFlex(std::ofstream &_out) {

    int n = 1;

    for (int i = 0; i < energyFunctions.size(); ++i) {
        ostringstream str1;
        str1 << n << ". " << energyFunctionsNameVec[i];
        _out << setw(fieldWidth) << str1.str();
        n += 1;
    }

    _out << endl;

}


void EnergyFunctionCalculatorStatistics::writeDataLineFlex(std::ofstream &_out, std::vector<double> &_energies) {

    for (int i = 0; i < _energies.size(); ++i) {
        _out << setw(fieldWidth) << _energies[i];
    }
    _out << endl;
}


void EnergyFunctionCalculatorStatistics::prepareGatherResultsFiles() {

    if (outputAcceptedSpinFlip) {
        ostringstream outSingleSpinFlipStreamNameAccepted;
        outSingleSpinFlipStreamNameAccepted << outFileCoreNameSpinFlips << "." << "accepted" << "." << "txt";
        outAccSpinFlip = new ofstream(outSingleSpinFlipStreamNameAccepted.str().c_str());

        *outAccSpinFlip << setw(fieldWidth) << "mcs";
        *outAccSpinFlip << setw(fieldWidth) << "pt";
        *outAccSpinFlip << setw(fieldWidth) << "prob";

        writeHeaderFlex(*outAccSpinFlip);
    }

    if (outputRejectedSpinFlip) {
        ostringstream outSingleSpinFlipStreamNameRejected;
        outSingleSpinFlipStreamNameRejected << outFileCoreNameSpinFlips << "." << "rejected" << "." << "txt";
        outRejSpinFlip = new ofstream(outSingleSpinFlipStreamNameRejected.str().c_str());

        *outRejSpinFlip << setw(fieldWidth) << "mcs";
        *outRejSpinFlip << setw(fieldWidth) << "pt";
        *outRejSpinFlip << setw(fieldWidth) << "prob";

        writeHeaderFlex(*outRejSpinFlip);
    }

    if (outputTotalSpinFlip) {
        ostringstream outSingleSpinFlipStreamNameTotal;
        outSingleSpinFlipStreamNameTotal << outFileCoreNameSpinFlips << "." << "total" << "." << "txt";
        outTotSpinFlip = new ofstream(outSingleSpinFlipStreamNameTotal.str().c_str());

        *outTotSpinFlip << setw(fieldWidth) << "mcs";
        *outTotSpinFlip << setw(fieldWidth) << "pt";
        *outTotSpinFlip << setw(fieldWidth) << "prob";

        writeHeaderFlex(*outTotSpinFlip);
    }

    gatherResultsFilesPrepared = true;

}

void EnergyFunctionCalculatorStatistics::outputResultsSingleSpinFlipGatherResults() {


    std::list < std::vector < double > > ::iterator
    lVecItr;
    std::list<bool>::iterator lItr;
    std::list<Point3D>::iterator flip_pt_litr;
    std::list<double>::iterator prob_list_litr;
    std::list<int>::iterator mcs_list_litr;


    lItr = accNotAccList.begin();
    flip_pt_litr = pixel_copy_attempt_points_list.begin();
    prob_list_litr = acceptance_probability_list.begin();
    mcs_list_litr = mcs_list.begin();

    for (lVecItr = totEnergyDataList.begin(); lVecItr != totEnergyDataList.end(); ++lVecItr) {

        vector<double> &energyData = *lVecItr;
        if (outputTotalSpinFlip)

            *outTotSpinFlip << setw(fieldWidth) << *mcs_list_litr;
        *outTotSpinFlip << setw(fieldWidth) << *flip_pt_litr;
        *outTotSpinFlip << setw(fieldWidth) << *prob_list_litr;

        writeDataLineFlex(*outTotSpinFlip, *lVecItr);

        if (*lItr) {//accepted flip
            if (outputAcceptedSpinFlip)
                *outAccSpinFlip << setw(fieldWidth) << *mcs_list_litr;
            *outAccSpinFlip << setw(fieldWidth) << *flip_pt_litr;
            *outAccSpinFlip << setw(fieldWidth) << *prob_list_litr;

            writeDataLineFlex(*outAccSpinFlip, *lVecItr);
        } else {//rejected flip
            if (outputRejectedSpinFlip)
                *outRejSpinFlip << setw(fieldWidth) << *mcs_list_litr;
            *outRejSpinFlip << setw(fieldWidth) << *flip_pt_litr;
            *outRejSpinFlip << setw(fieldWidth) << *prob_list_litr;
            writeDataLineFlex(*outRejSpinFlip, *lVecItr);
        }
        ++lItr;
        ++flip_pt_litr;
        ++prob_list_litr;
        ++mcs_list_litr;
    }


}


void EnergyFunctionCalculatorStatistics::outputResultsSingleSpinFlip() {
    ostringstream outSingleSpinFlipStreamNameAccepted;
    outSingleSpinFlipStreamNameAccepted << outFileCoreNameSpinFlips << "." << "accepted" << "." << mcs << "." << "txt";

    ostringstream outSingleSpinFlipStreamNameRejected;
    outSingleSpinFlipStreamNameRejected << outFileCoreNameSpinFlips << "." << "rejected" << "." << mcs << "." << "txt";

    ostringstream outSingleSpinFlipStreamNameTotal;
    outSingleSpinFlipStreamNameTotal << outFileCoreNameSpinFlips << "." << "total" << "." << mcs << "." << "txt";

    ofstream outSingleSpinFlipAccepted(outSingleSpinFlipStreamNameAccepted.str().c_str());
    ofstream outSingleSpinFlipRejected(outSingleSpinFlipStreamNameRejected.str().c_str());
    ofstream outSingleSpinFlipTotal(outSingleSpinFlipStreamNameTotal.str().c_str());

    writeHeaderFlex(outSingleSpinFlipAccepted);
    writeHeaderFlex(outSingleSpinFlipRejected);
    writeHeaderFlex(outSingleSpinFlipTotal);

    std::list < std::vector < double > > ::iterator
    lVecItr;
    std::list<bool>::iterator lItr;
    std::list<Point3D>::iterator flip_pt_litr;
    std::list<double>::iterator prob_list_litr;
    std::list<int>::iterator mcs_list_litr;

    mcs_list_litr = mcs_list.begin();
    flip_pt_litr = pixel_copy_attempt_points_list.begin();
    prob_list_litr = acceptance_probability_list.begin();
    lItr = accNotAccList.begin();


    for (lVecItr = totEnergyDataList.begin(); lVecItr != totEnergyDataList.end(); ++lVecItr) {

        vector<double> &energyData = *lVecItr;
        writeDataLineFlex(outSingleSpinFlipTotal, *lVecItr);
        if (*lItr) {//accepted flip

            outSingleSpinFlipAccepted << setw(fieldWidth) << *mcs_list_litr;
            outSingleSpinFlipAccepted << setw(fieldWidth) << *flip_pt_litr;
            outSingleSpinFlipAccepted << setw(fieldWidth) << *prob_list_litr;
            writeDataLineFlex(outSingleSpinFlipAccepted, *lVecItr);
        } else {//rejected flip
            outSingleSpinFlipRejected << setw(fieldWidth) << *mcs_list_litr;
            outSingleSpinFlipRejected << setw(fieldWidth) << *flip_pt_litr;
            outSingleSpinFlipRejected << setw(fieldWidth) << *prob_list_litr;
            writeDataLineFlex(outSingleSpinFlipRejected, *lVecItr);
        }
        ++lItr;
        ++flip_pt_litr;
        ++prob_list_litr;
        ++mcs_list_litr;
    }


}


void EnergyFunctionCalculatorStatistics::outputResults() {
    CC3D_Log(LOG_DEBUG) << "-------------ENERGY CALCULATOR STATISTICS-------------";
    CC3D_Log(LOG_DEBUG) << "Accepted Energy:";
    double totAccEnergyChange = 0;


    calculateStatData(); //this actually calculates stat data and allocates all necessary vectors

    for (int i = 0; i < energyFunctions.size(); ++i) {
        CC3D_Log(LOG_DEBUG) << "TOT "<<energyFunctionsNameVec[i]<<" "<<avgEnergyVectorTot[i]*NTot<<" avg: "<<avgEnergyVectorTot[i]<<" stdDev: "<<stdDevEnergyVectorTot[i];
        CC3D_Log(LOG_DEBUG) << "ACC "<<energyFunctionsNameVec[i]<<" "<<avgEnergyVectorAcc[i]*NAcc<<" avg: "<<avgEnergyVectorAcc[i]<<" stdDev: "<<stdDevEnergyVectorAcc[i];
        CC3D_Log(LOG_DEBUG) << "REJ "<<energyFunctionsNameVec[i]<<" "<<avgEnergyVectorRej[i]*NRej<<" avg: "<<avgEnergyVectorRej[i]<<" stdDev: "<<stdDevEnergyVectorRej[i];

        totAccEnergyChange += avgEnergyVectorAcc[i] * NAcc;
    }

    if (out) {
        (*out) << setw(fieldWidth) << mcs << setw(fieldWidth) << NAcc << setw(fieldWidth) << NRej << setw(fieldWidth)
               << NTot;
        for (int i = 0; i < energyFunctions.size(); ++i) {
            (*out) << setw(fieldWidth) << avgEnergyVectorAcc[i] << setw(fieldWidth) << stdDevEnergyVectorAcc[i];
        }
        for (int i = 0; i < energyFunctions.size(); ++i) {
            (*out) << setw(fieldWidth) << avgEnergyVectorRej[i] << setw(fieldWidth) << stdDevEnergyVectorRej[i];
        }
        for (int i = 0; i < energyFunctions.size(); ++i) {
            (*out) << setw(fieldWidth) << avgEnergyVectorTot[i] << setw(fieldWidth) << stdDevEnergyVectorTot[i];
        }
        (*out) << endl;

	}
	CC3D_Log(LOG_DEBUG) << "TOTAL ACC ENERGY CHANGE="<<totAccEnergyChange;
    CC3D_Log(LOG_DEBUG) << "-------------End of ENERGY CALCULATOR STATISTICS-------------";
    CC3D_Log(LOG_DEBUG) << "Output File name = "<<outFileName;

}

void EnergyFunctionCalculatorStatistics::init(CC3DXMLElement *_xmlData) {
    outFileName = "";
    outFileCoreNameSpinFlips = "";
    outputEverySpinFlip = false;
    gatherResultsSpinFlip = false;
    outputAcceptedSpinFlip = false;
    outputRejectedSpinFlip = false;
    outputTotalSpinFlip = false;
    analysisFrequency = 1;
    singleSpinFrequency = 1;


    CC3DXMLElement *outputFileNameElem = _xmlData->getFirstElement("OutputFileName");
    if (outputFileNameElem) {
        outFileName = outputFileNameElem->getText();
        out = new ofstream(outFileName.c_str());
        if (outputFileNameElem->findAttribute("Frequency"))
            analysisFrequency = outputFileNameElem->getAttributeAsUInt("Frequency");
    }

    CC3DXMLElement *outputCoreFileNameSpinFlipsElem = _xmlData->getFirstElement("OutputCoreFileNameSpinFlips");

    if (outputCoreFileNameSpinFlipsElem) {
        outFileCoreNameSpinFlips = outputCoreFileNameSpinFlipsElem->getText();
        outputEverySpinFlip = true;
        if (outputCoreFileNameSpinFlipsElem->findAttribute("Frequency"))
            singleSpinFrequency = outputCoreFileNameSpinFlipsElem->getAttributeAsUInt("Frequency");
        if (outputCoreFileNameSpinFlipsElem->findAttribute("GatherResults"))
            gatherResultsSpinFlip = true;
        if (outputCoreFileNameSpinFlipsElem->findAttribute("OutputAccepted"))
            outputAcceptedSpinFlip = true;
        if (outputCoreFileNameSpinFlipsElem->findAttribute("OutputRejected"))
            outputRejectedSpinFlip = true;
        if (outputCoreFileNameSpinFlipsElem->findAttribute("OutputTotal"))
            outputTotalSpinFlip = true;

	}
	CC3D_Log(LOG_DEBUG) << "outFileName="<<outFileName;
    CC3D_Log(LOG_DEBUG) << "outFileCoreNameSpinFlips="<<outFileCoreNameSpinFlips;
    CC3D_Log(LOG_DEBUG) << "outputEverySpinFlip="<<outputEverySpinFlip;
    CC3D_Log(LOG_DEBUG) << "gatherResultsSpinFlip="<<gatherResultsSpinFlip;
    CC3D_Log(LOG_DEBUG) << "outputAcceptedSpinFlip="<<outputAcceptedSpinFlip;
    CC3D_Log(LOG_DEBUG) << "outputRejectedSpinFlip="<<outputRejectedSpinFlip;
    CC3D_Log(LOG_DEBUG) << "outputTotalSpinFlip="<<outputTotalSpinFlip;
    CC3D_Log(LOG_DEBUG) << "analysisFrequency="<<analysisFrequency;
    CC3D_Log(LOG_DEBUG) << "singleSpinFrequency="<<singleSpinFrequency;


}
