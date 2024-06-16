#ifndef POTTSPARSEDATA_H
#define POTTSPARSEDATA_H

#include <CompuCell3D/ParseData.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Potts3D/CellTypeMotilityData.h>
#include <PublicUtilities/Units/Unit.h>
#include <CompuCell3D/CompuCellLibDLLSpecifier.h>
#include <vector>


namespace CompuCell3D {

    class EnergyFunctionCalculatorParseData;

    class EnergyFunctionCalculatorStatisticsParseData;

    class COMPUCELLLIB_EXPORT EnergyFunctionCalculatorParseData

    :public ParseData {
    public:

    EnergyFunctionCalculatorParseData(std::string
    _moduleName):
    ParseData(_moduleName)
            {}
    virtual ~

    EnergyFunctionCalculatorParseData() {}
};

class COMPUCELLLIB_EXPORT EnergyFunctionCalculatorStatisticsParseData

:public EnergyFunctionCalculatorParseData{
public:

EnergyFunctionCalculatorStatisticsParseData() : EnergyFunctionCalculatorParseData("Statistics") {
    outputEverySpinFlip = false;
    gatherResultsSpinFlip = false;
    outputAcceptedSpinFlip = false;
    outputRejectedSpinFlip = false;
    outputTotalSpinFlip = false;
    analysisFrequency = 1;
    singleSpinFrequency = 1;
}

virtual ~

EnergyFunctionCalculatorStatisticsParseData() {}

void OutputFileName(std::string _outFileName, unsigned int _analysisFrequency = 1) {
    outFileName = _outFileName;
    analysisFrequency = _analysisFrequency;
}

void OutputCoreFileNameSpinFlips(std::string _outFileCoreNameSpinFlips, unsigned int _singleSpinFrequency = 1) {
    outFileCoreNameSpinFlips = _outFileCoreNameSpinFlips;
    singleSpinFrequency = _singleSpinFrequency;
    outputEverySpinFlip = true;
}

void GatherResults(bool _gatherResultsSpinFlip) { gatherResultsSpinFlip = _gatherResultsSpinFlip; }

void OutputAccepted(bool _outputAcceptedSpinFlip) { outputAcceptedSpinFlip = _outputAcceptedSpinFlip; }

void OutputRejected(bool _outputRejectedSpinFlip) { outputRejectedSpinFlip = _outputRejectedSpinFlip; }

void OutputTotal(bool _outputTotalSpinFlip) { outputTotalSpinFlip = _outputTotalSpinFlip; }

std::string outFileName;
unsigned int analysisFrequency;
unsigned int singleSpinFrequency;
bool gatherResultsSpinFlip;
bool outputAcceptedSpinFlip;
bool outputRejectedSpinFlip;
bool outputTotalSpinFlip;
bool outputEverySpinFlip;
std::string outFileCoreNameSpinFlips;

};


class COMPUCELLLIB_EXPORT PottsParseData

:public ParseData{
public:

PottsParseData() :
        ParseData("Potts") {
    numSteps = 0;
    anneal = 0;
    flip2DimRatio = 1.;
    temperature = 0.;
    depth = 1.1;
    depthFlag = false;
    seed = 0;
    debugOutputFrequency = 1;
    latticeType = "square";
    dimensionType = "default";
    shapeFlag = false;
    shapeAlgorithm = "Default";
    acceptanceFunctionName = "Default";
    fluctuationAmplitudeFunctionName = "Min";
    shapeIndex = 0;
    shapeSize = 0;
    shapeInputfile = "none";
    shapeReg = "";
    offset = 0.;
    kBoltzman = 1.0;
    energyFcnParseDataPtr = 0;
    neighborOrder = 1;
//				massUnit=Unit("10^-15*kg");
//				lengthUnit=Unit("10^-6*m");
//				timeUnit=Unit("s");
}

virtual ~

PottsParseData() {
    if (energyFcnParseDataPtr) {
        delete energyFcnParseDataPtr;
        energyFcnParseDataPtr = 0;
    }
}

EnergyFunctionCalculatorStatisticsParseData *getEnergyFunctionCalculatorStatisticsParseData() {
    if (energyFcnParseDataPtr) {
        delete energyFcnParseDataPtr;
        energyFcnParseDataPtr = 0;
    }
    EnergyFunctionCalculatorStatisticsParseData * efcspdPtr = new EnergyFunctionCalculatorStatisticsParseData();
    energyFcnParseDataPtr = efcspdPtr;
    return efcspdPtr;


}

unsigned int numSteps;
unsigned int anneal;
double flip2DimRatio;
double temperature;
double depth;
bool depthFlag;
unsigned int seed;
unsigned int debugOutputFrequency;
std::string boundary_x;
std::string boundary_y;
std::string boundary_z;
std::string algorithmName;
std::string latticeType;
std::string dimensionType;
std::string acceptanceFunctionName;
std::string fluctuationAmplitudeFunctionName;
bool shapeFlag;
std::string shapeAlgorithm;
int shapeIndex;
int shapeSize;
std::string shapeInputfile;
std::string shapeReg;
double offset;
double kBoltzman;
unsigned int neighborOrder;
std::vector <CellTypeMotilityData> cellTypeMotilityVector;


Dim3D dim;
EnergyFunctionCalculatorParseData *energyFcnParseDataPtr;


//units
//			Unit massUnit;
//			Unit lengthUnit;
//			Unit timeUnit;

void Dimensions(Dim3D _dim) { dim = _dim; }

void Steps(unsigned int _numSteps) { numSteps = _numSteps; }

void Anneal(unsigned int _anneal) { anneal = _anneal; }

void FlipNeighborMaxDistance(double _depth) {
    depth = _depth;
    depthFlag = true;
}

void Flip2DimRatio(double _flip2DimRatio) { flip2DimRatio = _flip2DimRatio; }

void Temperature(double _temperature) { temperature = _temperature; }

void KBoltzman(double _kBoltzman) { kBoltzman = _kBoltzman; }

void Offset(double _offset) { offset = _offset; }

void NeighborOrder(double _neighborOrder) { neighborOrder = _neighborOrder; }

void LatticeType(std::string _latticeType) { latticeType = _latticeType; }

void AcceptanceFunctionName(std::string _acceptanceFunctionName) { acceptanceFunctionName = _acceptanceFunctionName; }

void FluctuationAmplitudeFunctionName(
        std::string _fluctuationAmplitudeFunctionName) { fluctuationAmplitudeFunctionName = _fluctuationAmplitudeFunctionName; }

void RandomSeed(unsigned int _seed) { seed = _seed; }

void DebugOutputFrequency(unsigned int _debugOutputFrequency) { debugOutputFrequency = _debugOutputFrequency; }

void Boundary_x(std::string _boundary_x) { boundary_x = _boundary_x; }

void Boundary_y(std::string _boundary_y) { boundary_y = _boundary_y; }

void Boundary_z(std::string _boundary_z) { boundary_z = _boundary_z; }

void MetropolisAlgorithm(std::string _algorithmName) { algorithmName = _algorithmName; }

void Shape(std::string _shapeAlgorithm, int _shapeIndex = 0, int _shapeSize = 0, std::string _shapeInputfile = "none",
           std::string _shapeReg = "") {
    shapeFlag = true;
    shapeAlgorithm = _shapeAlgorithm;
    shapeIndex = _shapeIndex;
    shapeSize = _shapeSize;
    shapeInputfile = _shapeInputfile;
    shapeReg = _shapeReg;

}

};


};
#endif
