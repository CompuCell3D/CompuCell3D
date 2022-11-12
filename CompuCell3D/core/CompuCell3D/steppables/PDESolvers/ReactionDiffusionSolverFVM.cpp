#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>
#include <CompuCell3D/plugins/CellType/CellTypePlugin.h>

#include <PublicUtilities/StringUtils.h>
#include <muParser/muParser.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <omp.h>
#include <Logger/CC3DLogger.h>

// macro to ensure CC3D_Log(LOG_DEBUG) << is enabled only when debuggig

using namespace std;
using namespace CompuCell3D;

/**
@author T.J. Sego, Ph.D.
*/

#include "ReactionDiffusionSolverFVM.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFVM::ReactionDiffusionSolverFVM()
	: DiffusableVector<float>(), lengthX(1.0), incTime(1.0)
{
	pUtils = 0;
	autoTimeSubStep = false;
	cellDataLoaded = false;
	integrationTimeStep = incTime;
	fluctuationCompensator = 0;
	fvMaxStableTimeSteps = 0;

	physTime = 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFVM::~ReactionDiffusionSolverFVM()
{
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr = 0;

	if(fvMaxStableTimeSteps) {
		delete fvMaxStableTimeSteps;
		fvMaxStableTimeSteps = 0;
	}

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { 
		delete concentrationFieldVector[fieldIndex];
		concentrationFieldVector[fieldIndex] = 0;
	}
}

void ReactionDiffusionSolverFVM::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


	CC3D_Log(LOG_DEBUG) << "*******************************";
	CC3D_Log(LOG_DEBUG) << "* Begin RDFVM initialization! *";
	CC3D_Log(LOG_DEBUG) << "*******************************";

	sim = _simulator;
	potts = _simulator->getPotts();
	automaton = potts->getAutomaton();
	xmlData = _xmlData;

	pUtils = sim->getParallelUtils();
	lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);

	// Get useful plugins
	CC3D_Log(LOG_DEBUG) << "Getting helpful plugins...";

	bool pluginAlreadyRegisteredFlag;

	//		Get cell type plugin
	CC3D_Log(LOG_DEBUG) << "   Cell type plugin...";

	CellTypePlugin *cellTypePlugin = (CellTypePlugin*)Simulator::pluginManager.get("CellType", &pluginAlreadyRegisteredFlag);
	ASSERT_OR_THROW("Cell type plugin must be registered for RDFVM, and in general.", pluginAlreadyRegisteredFlag);

	//		Get pixel tracker plugin
	
	pixelTrackerPlugin = (PixelTrackerPlugin*)Simulator::pluginManager.get("PixelTracker", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		CC3DXMLElement *pixelTrackerXML = sim->getCC3DModuleData("Plugin", "PixelTracker");
		pixelTrackerPlugin->init(sim, pixelTrackerXML);
	}
	
	// 		Get neighbor tracker plugin
	neighborTrackerPlugin = (NeighborTrackerPlugin*)Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		CC3DXMLElement *neighborTrackerXML = sim->getCC3DModuleData("Plugin", "NeighborTracker");
		neighborTrackerPlugin->init(sim, neighborTrackerXML);
	}

	fieldDim = potts->getCellFieldG()->getDim();

	boundaryStrategy = BoundaryStrategy::getInstance();

	maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

	// Get static inputs
	CC3D_Log(LOG_DEBUG) << "Getting static RDFVM Solver inputs...";

	//		Cell types

	CC3D_Log(LOG_DEBUG) << "Getting cell types...";

	std::map<unsigned char, std::string> typeNameMap = cellTypePlugin->getTypeNameMap();
	std::map<unsigned char, std::string>::iterator typeNameMap_itr;
	numCellTypes = 0;
	cellTypeNameToIndexMap.clear();
	for (typeNameMap_itr = typeNameMap.begin(); typeNameMap_itr != typeNameMap.end(); ++typeNameMap_itr) {
		cellTypeNameToIndexMap.insert(make_pair(typeNameMap_itr->second, (unsigned int)typeNameMap_itr->first));
		++numCellTypes;
	}

	// Currently disallow hex lattices
	ASSERT_OR_THROW("Hexagonal lattices are currently not supported by FVM Solver.", boundaryStrategy->getLatticeType() != HEXAGONAL_LATTICE);

	// Get solver inputs
	CC3D_Log(LOG_DEBUG)<<"Getting solver inputs...";
	CC3DXMLElement *el;

	//		Time discretization
	if (xmlData->findElement("DeltaT")) {
		el = xmlData->getFirstElement("DeltaT");
		incTime = (float)(el->getDouble());
		ASSERT_OR_THROW("FVM time increment must be greater than zero.", incTime > 0.0);
		if (el->findAttribute("unit")) { setUnitsTime(el->getAttribute("unit")); }

		CC3D_Log(LOG_DEBUG) << "   Got time discretization: " << incTime << " " << getUnitsTime() << "/step";

	}

	//		Spatial discretization
	//		If 2D and z-length not specified, use unit length for z
	float DeltaX = lengthX;
	float DeltaY = lengthX;
	float DeltaZ = 1.0;
	if (xmlData->findElement("DeltaX")) {
		DeltaX = (float)(xmlData->getFirstElement("DeltaX")->getDouble());

		CC3D_Log(LOG_DEBUG) << "   Got x-dimension discretization: " << DeltaX << " m";

		ASSERT_OR_THROW("FVM spatial discretization must be greater than zero.", DeltaX > 0.0);
		if (xmlData->findElement("DeltaY")) {
			DeltaY = (float)(xmlData->getFirstElement("DeltaY")->getDouble());
			ASSERT_OR_THROW("FVM spatial discretization must be greater than zero.", DeltaY > 0.0);
		}
		else { DeltaY = DeltaX; }

		CC3D_Log(LOG_DEBUG) << "   Got y-dimension discretization: " << DeltaY << " m";

		if (xmlData->findElement("DeltaZ")) {
			DeltaZ = (float)(xmlData->getFirstElement("DeltaZ")->getDouble());
			ASSERT_OR_THROW("FVM spatial discretization must be greater than zero.", DeltaZ > 0.0);
		}
		else if (maxNeighborIndex > 3) { DeltaZ = DeltaX; }

		CC3D_Log(LOG_DEBUG) << "   Got z-dimension discretization: " << DeltaZ << " m";

	}

	//		Diffusion fields

	CC3D_Log(LOG_DEBUG) << "Getting diffusion fields...";

	CC3DXMLElementList fieldXMLVec = _xmlData->getElements("DiffusionField");

	numFields = (unsigned int)fieldXMLVec.size();

	fieldNameToIndexMap.clear();
	concentrationFieldNameVector = std::vector<std::string>(numFields, "");
	concentrationFieldVector = std::vector<RDFVMField3DWrap<float> *>(numFields, 0);

	fieldSymbolsVec = std::vector<std::string>(numFields, "");
	fieldExpressionStringsDiag = std::vector<std::vector<std::string> >(numFields, std::vector<std::string>(0, ""));
	fieldExpressionStringsOffDiag = std::vector<std::vector<std::string> >(numFields, std::vector<std::string>(0, ""));
	std::vector<std::string> initialExpressionStrings = std::vector<std::string>(numFields, "");

	constantDiffusionCoefficientsVec = std::vector<double>(numFields, 0);
	diffusivityFieldIndexToFieldMap = std::vector<Field3D<float> *>(numFields, 0);
	diffusivityFieldInitialized = std::vector<bool>(numFields, false);
	constantDiffusionCoefficientsVecCellType = std::vector<std::vector<double> >(numFields, std::vector<double>(numCellTypes, 0.0));
	diffusivityModeInitializerPtrs = std::vector<DiffusivityModeInitializer>(numFields, DiffusivityModeInitializer(nullptr));

	constantPermeationCoefficientsVecCellType = std::vector<std::vector<std::vector<double> > >(numFields, 
		std::vector<std::vector<double> >(numCellTypes, 
			std::vector<double>(numCellTypes, 0.0)));
	constPermBiasCoeffsVecCellType = std::vector<std::vector<std::vector<double> > >(numFields, 
		std::vector<std::vector<double> >(numCellTypes, 
			std::vector<double>(numCellTypes, 1.0)));
	usingSimplePermeableInterfaces = std::vector<bool>(numFields, false);

	fluxConditionInitializerPtrs = std::vector<FluxConditionInitializer>(numFields, &ReactionDiffusionSolverFVM::useDiffusiveSurfaces);

	std::vector<bool> useConstantDiffusivityBool = std::vector<bool>(numFields, false);
	std::vector<bool> useConstantDiffusivityByTypeBool = std::vector<bool>(numFields, false);
	std::vector<bool> useFieldDiffusivityInMediumBool = std::vector<bool>(numFields, false);
	std::vector<bool> useFieldDiffusivityEverywhereBool = std::vector<bool>(numFields, false);

	std::map<std::string, CC3DXMLElement *> bcElementCollector; 
	bcElementCollector.clear();

	std::vector<std::string> fieldInitialExpr = std::vector<std::string>(numFields, "");
	std::vector<bool> useFieldInitialExprBool = std::vector<bool>(numFields, false);
	secrFieldVec.clear();
	secrFieldVec.reserve(numFields);

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		el = fieldXMLVec[fieldIndex];
		ASSERT_OR_THROW("Each diffusion field must be given a name with the DiffusionField attribute Name", el->findAttribute("Name"));
		std::string fieldName = el->getAttribute("Name");

		CC3D_Log(LOG_DEBUG)<<"   Got field name: " << fieldName;

		// Check duplicates
		std::vector<std::string>::iterator fieldNameVec_itr = find(concentrationFieldNameVector.begin(), concentrationFieldNameVector.end(), fieldName);
		ASSERT_OR_THROW("Each FVM diffusion field must have a unique name", fieldNameVec_itr == concentrationFieldNameVector.end());

		CC3D_Log(LOG_DEBUG) << "   Generating field wrap...";
		
		fieldNameToIndexMap.insert(make_pair(fieldName, fieldIndex));
		concentrationFieldNameVector[fieldIndex] = fieldName;
		concentrationFieldVector[fieldIndex] = new RDFVMField3DWrap<float>(this, fieldName);

		CC3D_Log(LOG_DEBUG) << "   Registering field with Simulator...";

		sim->registerConcentrationField(fieldName, concentrationFieldVector[fieldIndex]);

		CC3DXMLElement *dData;
		CC3DXMLElement *dDataEl;
		
		// Diffusion data

		CC3D_Log(LOG_DEBUG) << "   Getting diffusion data...";

		ASSERT_OR_THROW("A DiffusionData element must be defined per FVM diffusion field", el->findElement("DiffusionData"));
		bool diffusionDefined = false;
		dData = el->getFirstElement("DiffusionData");

		//		Load medium diffusivity and mode if present
		dDataEl = dData->getFirstElement("DiffusionConstant");
		if (dDataEl) {
			diffusionDefined = true;
			constantDiffusionCoefficientsVec[fieldIndex] = dDataEl->getDouble();

			CC3D_Log(LOG_DEBUG) << "   Got diffusion constant: " << constantDiffusionCoefficientsVec[fieldIndex] << " m2/s";

			useConstantDiffusivityBool[fieldIndex] = true;
		}
		//		Load diffusivity by type
		if (dData->findElement("DiffusivityByType")) {
			useConstantDiffusivityBool[fieldIndex] = false;
			useConstantDiffusivityByTypeBool[fieldIndex] = true;
			CC3D_Log(LOG_DEBUG) << "   Got diffusivity by type.";
		}
		//		Load diffusivity field in medium if present
		if (dData->findElement("DiffusivityFieldInMedium")) {
			if (diffusionDefined) { 
				CC3D_Log(LOG_DEBUG) << "Warning: duplicate diffusion mode. Overwriting with DiffusivityFieldInMedium" ;
				}
			else { 
				CC3D_Log(LOG_DEBUG) << "   Got diffusivity field in medium. Diffusivity field is named: " + fieldName + diffusivityFieldSuffixStd;
				 }
			diffusionDefined = true;
			useConstantDiffusivityBool[fieldIndex] = false;
			useConstantDiffusivityByTypeBool[fieldIndex] = false;
			useFieldDiffusivityInMediumBool[fieldIndex] = true;
		}
		//		Load diffusivity field everywhere if present
		if (dData->findElement("DiffusivityFieldEverywhere")) {
			if (diffusionDefined) { 
				CC3D_Log(LOG_DEBUG) << "Warning: duplicate diffusion mode. Overwriting with DiffusivityFieldEverywhere";
				 }
			else { 
				CC3D_Log(LOG_DEBUG) << "   Got diffusivity field everywhere. Diffusivity field is named: " + fieldName + diffusivityFieldSuffixStd;
				}
			diffusionDefined = true;
			useConstantDiffusivityBool[fieldIndex] = false;
			useConstantDiffusivityByTypeBool[fieldIndex] = false;
			useFieldDiffusivityInMediumBool[fieldIndex] = false;
			useFieldDiffusivityEverywhereBool[fieldIndex] = true;
		}
		if (dData->findElement("InitialConcentrationExpression")) {
			initialExpressionStrings[fieldIndex] = dData->getFirstElement("InitialConcentrationExpression")->getText();
			CC3D_Log(LOG_DEBUG) << "   Got initial concentration expression: " + initialExpressionStrings[fieldIndex];
		}
		ASSERT_OR_THROW("A diffusion mode must be defined in DiffusionData.", diffusionDefined);

		//		Initialize cell type diffusivity coefficients as the same as for the field before loading type specifications
		constantDiffusionCoefficientsVecCellType[fieldIndex] = std::vector<double>(numCellTypes, constantDiffusionCoefficientsVec[fieldIndex]);

		//		Load all present cell type diffusion data, for future ref.
		for (CC3DXMLElement *typeData : dData->getElements("DiffusionCoefficient")) {

			std::string cellTypeName = typeData->getAttribute("CellType");
			std::map<std::string, unsigned int>::iterator cellTypeNameToIndexMap_itr = cellTypeNameToIndexMap.find(cellTypeName);
			if (cellTypeNameToIndexMap_itr != cellTypeNameToIndexMap.end()) {
				double typeDiffC = typeData->getDouble();
				constantDiffusionCoefficientsVecCellType[fieldIndex][cellTypeNameToIndexMap_itr->second] = typeDiffC;
				CC3D_Log(LOG_DEBUG) << "   Got cell type (" << cellTypeName << ") diffusivity: " << typeDiffC << " m2/s";
			}
		}

		//		Load all present cell type permeability data, for future ref.
		//			Interface permeation coefficients
		for (CC3DXMLElement *typeData : dData->getElements("PermIntCoefficient")) {

			std::string cellType1Name = typeData->getAttribute("Type1");
			std::string cellType2Name = typeData->getAttribute("Type2");
			std::map<std::string, unsigned int>::iterator cellType1NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType1Name);
			std::map<std::string, unsigned int>::iterator cellType2NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType2Name);
			if (cellType1NameToIndexMap_itr != cellTypeNameToIndexMap.end() && cellType2NameToIndexMap_itr != cellTypeNameToIndexMap.end()) {
				double permC = typeData->getDouble();
				constantPermeationCoefficientsVecCellType[fieldIndex][cellType1NameToIndexMap_itr->second][cellType2NameToIndexMap_itr->second] = permC;
				constantPermeationCoefficientsVecCellType[fieldIndex][cellType2NameToIndexMap_itr->second][cellType1NameToIndexMap_itr->second] = permC;
				CC3D_Log(LOG_DEBUG) << "   Got cell type (" << cellType1Name << ", " << cellType2Name << ") interface permeation coefficient: " << permC << " m/s";
			}
		}
		//			Interface bias coefficients
		for (CC3DXMLElement *typeData : dData->getElements("PermIntBias")) {

			std::string cellType1Name = typeData->getAttribute("Type1");
			std::string cellType2Name = typeData->getAttribute("Type2");
			std::map<std::string, unsigned int>::iterator cellType1NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType1Name);
			std::map<std::string, unsigned int>::iterator cellType2NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType2Name);
			if (cellType1NameToIndexMap_itr != cellTypeNameToIndexMap.end() && cellType2NameToIndexMap_itr != cellTypeNameToIndexMap.end()) {
				double biasC = typeData->getDouble();
				constPermBiasCoeffsVecCellType[fieldIndex][cellType1NameToIndexMap_itr->second][cellType2NameToIndexMap_itr->second] = biasC;
				CC3D_Log(LOG_DEBUG) <<  "   Got cell type (" << cellType1Name << ", " << cellType2Name << ") interface bias coefficient: " << biasC;
			}
		}

		//		Load simple permeable membranes if present
		usingSimplePermeableInterfaces[fieldIndex] = dData->findElement("SimplePermInt");
		if (usingSimplePermeableInterfaces[fieldIndex]) { fluxConditionInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::usePermeableSurfaces; }

		//		Load initial field expression if present
		useFieldInitialExprBool[fieldIndex] = dData->findElement("InitialConcentrationExpression");
		if (useFieldInitialExprBool[fieldIndex]) { fieldInitialExpr[fieldIndex] = dData->getFirstElement("InitialConcentrationExpression")->getData(); }

		// Reaction data
		CC3D_Log(LOG_DEBUG)<<"   Getting reaction data...";
		fieldExpressionStringsDiag[fieldIndex].clear();
		fieldExpressionStringsOffDiag[fieldIndex].clear();

		CC3DXMLElement *rData = el->getFirstElement("ReactionData");
		CC3DXMLElement *rDataEl;
		if (rData) {
			rDataEl = rData->getFirstElement("ExpressionSymbol");
			if (rDataEl) {
				fieldSymbolsVec[fieldIndex] = rDataEl->getText();
				CC3D_Log(LOG_DEBUG) <<  "   Got reaction expression symbol: " << fieldSymbolsVec[fieldIndex];
			}

			if (rData->findElement("ExpressionMult")){
				for (CC3DXMLElement *expData : rData->getElements("ExpressionMult")) {

					std::string expStr = expData->getData();
					fieldExpressionStringsDiag[fieldIndex].push_back(expStr);
					CC3D_Log(LOG_DEBUG) <<" Got multiplier reaction expression: " << expStr;
				}
			}

			if (rData->findElement("ExpressionIndep")) {
				for (CC3DXMLElement *expData : rData->getElements("ExpressionIndep")) {

					std::string expStr = expData->getData();
					fieldExpressionStringsOffDiag[fieldIndex].push_back(expStr);
					CC3D_Log(LOG_DEBUG) << "   Got independent reaction expression: " << expStr ;
				}
			}
		}

		// Secretion data

		SecretionData secrData;
		if(el->findElement("SecretionData")) {
			secrData.update(el->getFirstElement("SecretionData"));
			secrData.initialize(automaton);
		}
		secrFieldVec.push_back(secrData);

		// Collect boundary conditions
		CC3D_Log(LOG_DEBUG) << "   Collecting boundary conditions...";

		CC3DXMLElement *bcData = el->getFirstElement("BoundaryConditions");
		if (bcData) { bcElementCollector.insert(make_pair(fieldName, bcData)); }
		
	}

	//		Assign reaction expression symbols for any field not already defined, for future ref
	for (unsigned int fieldIndex = 0; fieldIndex < fieldSymbolsVec.size(); ++fieldIndex) {
		if (fieldSymbolsVec[fieldIndex].size() == 0) {
			std::string fieldName = concentrationFieldNameVector[fieldIndex];
			fieldSymbolsVec[fieldIndex] = fieldName + expressionSuffixStd;
			CC3D_Log(LOG_DEBUG) <<  "   Assigning reaction expression symbol for " << fieldName << ": " << fieldSymbolsVec[fieldIndex];
		}
	}

	// Load diffusion initializers
	CC3D_Log(LOG_DEBUG) <<  "Loading diffusion initializers..." ;

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		CC3D_Log(LOG_DEBUG) << "   " << concentrationFieldNameVector[fieldIndex] << ": ";

		if (useConstantDiffusivityBool[fieldIndex]) {
			CC3D_Log(LOG_DEBUG) <<  "constant diffusivity.";
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useConstantDiffusivity;
		}
		else if (useConstantDiffusivityByTypeBool[fieldIndex]) {
			CC3D_Log(LOG_DEBUG) << "constant diffusivity by type.";
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useConstantDiffusivityByType;
		}
		else if (useFieldDiffusivityInMediumBool[fieldIndex]) {
			CC3D_Log(LOG_DEBUG) << "diffusivity field in medium.";
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useFieldDiffusivityInMedium;
		}
		else if (useFieldDiffusivityEverywhereBool[fieldIndex]) {
			CC3D_Log(LOG_DEBUG) << "diffusivity field everywhere.";
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useFieldDiffusivityEverywhere;
		}
	}
	
	// Build surface mappings
	// Note: will need updated for hex lattices
	CC3D_Log(LOG_DEBUG) << "Building surface mappings..." ;

	indexMapSurfToCoord = std::vector<unsigned int>(maxNeighborIndex, 0);
	surfaceNormSign = std::vector<int>(maxNeighborIndex, 0);
	surfaceMapNameToIndex.clear();

	//		Standards, for handling inconsistent data from boundary strategy
	std::vector<std::string> surfNameStd = std::vector<std::string>{ "MaxX", "MinX", "MaxY", "MinY" , "MaxZ", "MinZ" };
	std::vector<unsigned int> indexMapSurfToCoordStd = std::vector<unsigned int>{ 0, 0, 1, 1, 2, 2 };
	std::vector<int> surfaceNormSignStd = std::vector<int>{ 1, -1, 1, -1, 1, -1 };

	std::vector<Point3D> offsetVec = boundaryStrategy->getOffsetVec();
	for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
		Point3D offset = offsetVec[nIdx];
		CC3D_Log(LOG_DEBUG) << "   Processing surface for neighbor relative offset (" << offset.x << ", " << offset.y << ", " << offset.z << ") -> ";
		if (offset.x > 0) {
			CC3D_Log(LOG_DEBUG) <<  "+x: " << nIdx ;

			indexMapSurfToCoord[nIdx] = 0;
			surfaceNormSign[nIdx] = 1;
			surfaceMapNameToIndex.insert(make_pair("MaxX", nIdx));
		}
		else if (offset.x < 0) {
			CC3D_Log(LOG_DEBUG) << "-x: " << nIdx;

			indexMapSurfToCoord[nIdx] = 0;
			surfaceNormSign[nIdx] = -1;
			surfaceMapNameToIndex.insert(make_pair("MinX", nIdx));
		}
		else if (offset.y > 0) {
			CC3D_Log(LOG_DEBUG) << "+y: " << nIdx;

			indexMapSurfToCoord[nIdx] = 1;
			surfaceNormSign[nIdx] = 1;
			surfaceMapNameToIndex.insert(make_pair("MaxY", nIdx));
		}
		else if (offset.y < 0) {
			CC3D_Log(LOG_DEBUG) << "-y: " << nIdx;

			indexMapSurfToCoord[nIdx] = 1;
			surfaceNormSign[nIdx] = -1;
			surfaceMapNameToIndex.insert(make_pair("MinY", nIdx));
		}
		else if (offset.z > 0) {
			CC3D_Log(LOG_DEBUG) << "+z: " << nIdx;

			indexMapSurfToCoord[nIdx] = 2;
			surfaceNormSign[nIdx] = 1;
			surfaceMapNameToIndex.insert(make_pair("MaxZ", nIdx));
		}
		else if (offset.z < 0) {
			CC3D_Log(LOG_DEBUG) <<  "-z: " << nIdx;

			indexMapSurfToCoord[nIdx] = 2;
			surfaceNormSign[nIdx] = -1;
			surfaceMapNameToIndex.insert(make_pair("MinZ", nIdx));
		}
		else { // Assume an order
			CC3D_Log(LOG_DEBUG) << "Warning: assuming a neighbor surface map: " << nIdx;

			indexMapSurfToCoord[nIdx] = indexMapSurfToCoordStd[nIdx];
			surfaceNormSign[nIdx] = surfaceNormSignStd[nIdx];
			surfaceMapNameToIndex.insert(make_pair(surfNameStd[nIdx], nIdx));
		}
	}
	setLengths(DeltaX, DeltaY, DeltaZ);

	// Load boundary conditions
	CC3D_Log(LOG_DEBUG) <<  "Loading boundary conditions...";

	periodicBoundaryCheckVector = std::vector<bool>(3, false);
	std::vector<tuple<std::string, float> > basicBCDataFieldTemplate = std::vector<tuple<std::string, float> >(6, tuple<std::string, float>("ConstantDerivative", 0.0));
	basicBCData = std::vector<std::vector<tuple<std::string, float> > >(numFields, basicBCDataFieldTemplate);
	std::string boundaryName;
	boundaryName = potts->getBoundaryXName();
	if (boundaryName == "periodic") {
		CC3D_Log(LOG_DEBUG) << "   Periodic x from Potts.";

		periodicBoundaryCheckVector[0] = true;
		basicBCDataFieldTemplate[getSurfaceIndexByName("MaxX")] = tuple<std::string, float>("Periodic", 0.0);
		basicBCDataFieldTemplate[getSurfaceIndexByName("MinX")] = tuple<std::string, float>("Periodic", 0.0);
	}
	boundaryName = potts->getBoundaryYName();
	if (boundaryName == "periodic") {
		CC3D_Log(LOG_DEBUG) << "   Periodic y from Potts.";

		periodicBoundaryCheckVector[1] = true;
		basicBCDataFieldTemplate[getSurfaceIndexByName("MaxY")] = tuple<std::string, float>("Periodic", 0.0);
		basicBCDataFieldTemplate[getSurfaceIndexByName("MinY")] = tuple<std::string, float>("Periodic", 0.0);
	}
	if (maxNeighborIndex > 3) {
		boundaryName = potts->getBoundaryZName();
		if (boundaryName == "periodic") {
			CC3D_Log(LOG_DEBUG) <<  "   Periodic z from Potts.";

			periodicBoundaryCheckVector[2] = true;
			basicBCDataFieldTemplate[getSurfaceIndexByName("MaxZ")] = tuple<std::string, float>("Periodic", 0.0);
			basicBCDataFieldTemplate[getSurfaceIndexByName("MinZ")] = tuple<std::string, float>("Periodic", 0.0);
		}
	}

	std::map<std::string, CC3DXMLElement *>::iterator bc_itr;
	for (unsigned int fieldIndex = 0; fieldIndex < concentrationFieldNameVector.size(); ++fieldIndex) {
		std::string fieldName = concentrationFieldNameVector[fieldIndex];
		std::vector<tuple<std::string, float> > basicBCDataField = basicBCDataFieldTemplate;
		CC3D_Log(LOG_DEBUG) << "Loading boundary conditions for field " + fieldName;

		bc_itr = bcElementCollector.find(fieldName);
		if (bc_itr != bcElementCollector.end()) {
			CC3DXMLElementList bcPlaneElementList = bc_itr->second->getElements("Plane");
			for (CC3DXMLElement *bcEl : bcPlaneElementList) {

				std::string axisString = bcEl->getAttribute("Axis");
				for (CC3DXMLElement *bcElSpec : bcEl->children) {

					std::string posString = bcElSpec->getAttribute("PlanePosition");
					std::string surfaceName = posString + axisString;
					unsigned int surfaceIndex = getSurfaceIndexByName(surfaceName);
					unsigned int dimIndex = getIndexSurfToCoord(surfaceIndex);
					ASSERT_OR_THROW("Cannot specify a boundary condition for a periodic boundary.", !periodicBoundaryCheckVector[dimIndex]);

					std::string bcTypeName = bcElSpec->getName();
					ASSERT_OR_THROW(std::string("Unknown boundary condition type: " + bcTypeName + ". Valid inputs are ConstantValue and ConstantDerivative"), 
						bcTypeName == "ConstantValue" || bcTypeName == "ConstantDerivative");
					float bcVal = (float)(bcElSpec->getAttributeAsDouble("Value"));
					CC3D_Log(LOG_DEBUG) << "   Got boundary condition " << bcTypeName << " for " << surfaceName << " with value " << bcVal;
					CC3D_Log(LOG_DEBUG) << "      Loading to surface index " << surfaceIndex;
					CC3D_Log(LOG_DEBUG) <<  " for dimension index " << dimIndex;
					
					basicBCDataField[surfaceIndex] = tuple<std::string, float>(bcTypeName, bcVal);
				}
			}
		}
		basicBCData[fieldIndex] = basicBCDataField;
	}

	// Initialize reaction expressions
	CC3D_Log(LOG_DEBUG) << "Initializing reaction expressions...";

	fieldExpressionStringsMergedDiag = std::vector<std::string>(numFields, "");
	fieldExpressionStringsMergedOffDiag = std::vector<std::string>(numFields, "");
	
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		CC3D_Log(LOG_DEBUG) << "Constructing reaction expressions for " + concentrationFieldNameVector[fieldIndex];

		std::string expDiag = "";
		std::string expOffDiag = "";

		if (fieldExpressionStringsDiag[fieldIndex].size() == 0) { expDiag = "0.0"; }
		else {
			expDiag = fieldExpressionStringsDiag[fieldIndex][0];
			if (fieldExpressionStringsDiag[fieldIndex].size() > 1) {
				for (unsigned int expIndex = 1; expIndex < fieldExpressionStringsDiag[fieldIndex].size(); ++expIndex) {
					expDiag += "+" + fieldExpressionStringsDiag[fieldIndex][expIndex];
				}
			}
			CC3D_Log(LOG_DEBUG) << "   Multiplier function: " + expDiag;
		}

		if (fieldExpressionStringsOffDiag[fieldIndex].size() == 0) { expOffDiag = "0.0"; }
		else {
			expOffDiag = fieldExpressionStringsOffDiag[fieldIndex][0];
			if (fieldExpressionStringsOffDiag[fieldIndex].size() > 1) {
				for (unsigned int expIndex = 1; expIndex < fieldExpressionStringsOffDiag[fieldIndex].size(); ++expIndex) {
					expOffDiag += "+" + fieldExpressionStringsOffDiag[fieldIndex][expIndex];
				}
			}
			CC3D_Log(LOG_DEBUG) << "   Independent function: " + expOffDiag;
		}

		if (expDiag.size() == 0) { expDiag = "0.0"; }
		if (expOffDiag.size() == 0) { expOffDiag = "0.0"; }
		fieldExpressionStringsMergedDiag[fieldIndex] = expDiag;
		fieldExpressionStringsMergedOffDiag[fieldIndex] = expOffDiag;
	}

	// Build lattice

	CC3D_Log(LOG_DEBUG) << "Building lattice...";

	initializeFVs(fieldDim);

	// Initialize concentrations

	CC3D_Log(LOG_DEBUG) << "Initializing concentrations...";

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
		if (useFieldInitialExprBool[fieldIndex]) { initializeFieldUsingEquation(fieldIndex, fieldInitialExpr[fieldIndex]); }

	//		Auto time stepping
	fvMaxStableTimeSteps = 0;
	autoTimeSubStep = _xmlData->findElement("AutoTimeSubStep");
	if (autoTimeSubStep) {
		CC3D_Log(LOG_DEBUG) << "RDVFM got automatic time sub-stepping.";
		fvMaxStableTimeSteps = new std::vector<double>(fieldDim.x * fieldDim.y * fieldDim.z, 0.0);
	}

    // 		FluctuationCompensator support
	if (_xmlData->findElement("FluctuationCompensator")) {

        fluctuationCompensator = new FluctuationCompensator(sim);

        for (unsigned int i = 0; i < concentrationFieldNameVector.size(); ++i)
            fluctuationCompensator->loadFieldName(concentrationFieldNameVector[i]);

        fluctuationCompensator->loadFields();

    }

	CC3D_Log(LOG_DEBUG) << "Registering RDFVM Solver...";

	potts->getCellFactoryGroupPtr()->registerClass(&ReactionDiffusionSolverFVMCellDataAccessor);
	sim->registerSteerableObject(this);

	CC3D_Log(LOG_DEBUG) << "*****************************";
	CC3D_Log(LOG_DEBUG) << "* End RDFVM initialization! *";
	CC3D_Log(LOG_DEBUG) << "*****************************";

}

void ReactionDiffusionSolverFVM::extraInit(Simulator *simulator) { }

// Not yet tested!
void ReactionDiffusionSolverFVM::handleEvent(CC3DEvent & _event) {

	//if (_event.id == LATTICE_RESIZE) {

		pUtils->setLock(lockPtr);

		std::vector<ReactionDiffusionSolverFV *> fvs = std::vector<ReactionDiffusionSolverFV *>((int)(fieldDim.x*fieldDim.y*fieldDim.z));

		fvs = fieldFVs;
		Dim3D fieldDimOld = fieldDim;

		CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);
		fieldDim = ev.newDim;
		initializeFVs(fieldDim);
		pUtils->getNumberOfProcessors();
		
		#pragma omp parallel for shared(fvs)
		for (int i=0;i<fvs.size();i++){
			Point3D pt = fvs[i]->getCoords();
			Point3D ptNew = pt;
			ptNew.x += ev.shiftVec.x;
			ptNew.y += ev.shiftVec.y;
			ptNew.z += ev.shiftVec.z;
			getFieldFV(ptNew)->setConcentrationVec(fvs[i]->getConcentrationVec());
		}

		if (fluctuationCompensator) {
			fluctuationCompensator->resetCorrections();
		}

		pUtils->unsetLock(lockPtr);
		
		update(xmlData, false);
	//}

}

/////////////////////////////////////////////////////////// Steppable interface //////////////////////////////////////////////////////////

void ReactionDiffusionSolverFVM::step(const unsigned int _currentStep) {

	pUtils->setLock(lockPtr);

	// Load cell data just in time if necessary
	if (!cellDataLoaded) { loadCellData(); }

	CC3D_Log(LOG_DEBUG) << "RDFVM Step begin...";
	
	auto &_fieldFVs = fieldFVs;

	if (fluctuationCompensator) fluctuationCompensator->applyCorrections();

	CC3D_Log(LOG_DEBUG) << "   Explicit RD integration...";

	double intTime = 0.0;

	fieldDim = potts->getCellFieldG()->getDim();
	auto _fieldDim = fieldDim;
	auto _secrFieldVec = this->secrFieldVec;

	while (intTime < incTime) {

		if (autoTimeSubStep) {

			CC3D_Log(LOG_DEBUG) <<  "      Integrating with maximum stable time step... ";

			#pragma omp parallel for shared (_fieldFVs, _secrFieldVec)
			for(int i = 0; i < _fieldFVs.size(); i++)
				_fieldFVs[i]->secrete(_secrFieldVec);



			#pragma omp parallel for shared (_fieldDim)
			for (int fieldIndex=0;fieldIndex<_fieldDim.x*_fieldDim.y*_fieldDim.z;fieldIndex++){
				fvMaxStableTimeSteps->at(fieldIndex) = this->getFieldFV(fieldIndex)->solveStable();
			}

			CC3D_Log(LOG_DEBUG) << "calculating maximum stable time step... ";

			// Might be more efficient using a combinable
			integrationTimeStep = min(*min_element(fvMaxStableTimeSteps->begin(), fvMaxStableTimeSteps->end()), incTime - intTime);
		}
		else { 

			CC3D_Log(LOG_DEBUG) << "      Integrating with fixed time step... ";

			#pragma omp parallel for shared (_fieldFVs, _secrFieldVec)
			for(int i = 0; i < _fieldFVs.size(); i++)
				_fieldFVs[i]->secrete(_secrFieldVec);

			integrationTimeStep = incTime - intTime;

			#pragma omp parallel for shared (_fieldFVs)
			for (int i=0;i< _fieldFVs.size();i++){
				_fieldFVs[i]->solve();
			}

		}

		CC3D_Log(LOG_DEBUG) << integrationTimeStep << " s." ;

		CC3D_Log(LOG_DEBUG) << "      Updating... ";
		
		#pragma omp parallel for shared (_fieldFVs)
		for (int i=0;i<_fieldFVs.size();i++){
			_fieldFVs[i]->update(this->getIntegrationTimeStep());
		}

		intTime += integrationTimeStep;
		physTime += integrationTimeStep;

		CC3D_Log(LOG_DEBUG) << "done: " << physTime / unitTimeConv << " " << getUnitsTime();

	}

	if (fluctuationCompensator) fluctuationCompensator->resetCorrections();

	CC3D_Log(LOG_DEBUG) << "RDFVM Step complete.";

	pUtils->unsetLock(lockPtr);

}

////////////////////////////////////////////////////////// Steerable interface ///////////////////////////////////////////////////////////
void ReactionDiffusionSolverFVM::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

	// Get steerable inputs
	//		Auto time stepping
	autoTimeSubStep = _xmlData->findElement("AutoTimeSubStep");
	if (autoTimeSubStep) {
		CC3D_Log(LOG_DEBUG) << "RDVFM got automatic time sub-stepping.";
	}

}

std::string ReactionDiffusionSolverFVM::steerableName() {
	return toString();
}

std::string ReactionDiffusionSolverFVM::toString() {
	return "ReactionDiffusionSolverFVM";
}

///////////////////////////////////////////////////////////// Solver routines ////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFVM::loadCellData() {

	CC3D_Log(LOG_DEBUG) << "RDFVM Initializing cell data...";

	initializeCellData(numFields);
	CC3D_Log(LOG_DEBUG) << "RDFVM Loading cell data...";

	setCellDiffusivityCoefficients();
	setCellPermeableCoefficients();

	cellDataLoaded = true;
}

void ReactionDiffusionSolverFVM::loadFieldExpressions() {
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		loadFieldExpressionMultiplier(fieldIndex);
		loadFieldExpressionIndependent(fieldIndex);
	}
}

void ReactionDiffusionSolverFVM::loadFieldExpressionMultiplier(unsigned int _fieldIndex) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		loadFieldExpressionMultiplier(_fieldIndex, _fieldFVs[i]);
	}
}

void ReactionDiffusionSolverFVM::loadFieldExpressionMultiplier(std::string _fieldName, std::string _expr) {
	setFieldExpressionMultiplier(_fieldName, _expr);
	loadFieldExpressionMultiplier(getFieldIndexByName(_fieldName));
}

void ReactionDiffusionSolverFVM::loadFieldExpressionIndependent(unsigned int _fieldIndex) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
				loadFieldExpressionIndependent(_fieldIndex, _fieldFVs[i]);
		}
}

void ReactionDiffusionSolverFVM::loadFieldExpressionIndependent(std::string _fieldName, std::string _expr) {
	setFieldExpressionIndependent(_fieldName, _expr);
	loadFieldExpressionIndependent(getFieldIndexByName(_fieldName));
}

void ReactionDiffusionSolverFVM::loadFieldExpressionMultiplier(unsigned int _fieldIndex, ReactionDiffusionSolverFV *_fv) {
	_fv->setDiagonalFunctionExpression(_fieldIndex, fieldExpressionStringsMergedDiag[_fieldIndex]);
}

void ReactionDiffusionSolverFVM::loadFieldExpressionIndependent(unsigned int _fieldIndex, ReactionDiffusionSolverFV *_fv) {
	_fv->setOffDiagonalFunctionExpression(_fieldIndex, fieldExpressionStringsMergedOffDiag[_fieldIndex]);
}

void ReactionDiffusionSolverFVM::initializeFVs(Dim3D _fieldDim) {
	// Generate finite volumes

    fieldFVs = std::vector<ReactionDiffusionSolverFV*>(_fieldDim.x*_fieldDim.y*_fieldDim.z);

	CC3D_Log(LOG_DEBUG) << "Constructing lattice with " << _fieldDim.x*_fieldDim.y*_fieldDim.z << " sites...";


    #pragma omp parallel for shared (_fieldDim)
	for (int ind=0;ind<_fieldDim.x*_fieldDim.y*_fieldDim.z;ind++){
		ReactionDiffusionSolverFV *fv = new ReactionDiffusionSolverFV(this, ind2pt(ind), (int)this->getConcentrationFieldNameVector().size());
		this->setFieldFV(ind, fv);
	}

	CC3D_Log(LOG_DEBUG) << "Initializing FVs...";

	auto &_fieldFVs = fieldFVs;

    #pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i< _fieldFVs.size();i++){
		_fieldFVs[i]->initialize();
	}
	CC3D_Log(LOG_DEBUG) << "Setting field symbols...";

	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
	for (int fieldIndex=0;fieldIndex<_fieldFVs[i]->getConcentrationVec().size();++fieldIndex){
			_fieldFVs[i]->registerFieldSymbol(fieldIndex, this->getFieldSymbol(fieldIndex));
		}
	}

	CC3D_Log(LOG_DEBUG) << "Loading field expressions...";

	loadFieldExpressions();

	CC3D_Log(LOG_DEBUG) << "Setting initial FV diffusivity method...";

	// Diffusion mode initializations
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { (this->*diffusivityModeInitializerPtrs[fieldIndex])(fieldIndex); }

	// Lattice-wide flux condition initializations
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { (this->*fluxConditionInitializerPtrs[fieldIndex])(fieldIndex); }

	// Apply basic boundary conditions

	CC3D_Log(LOG_DEBUG) << "Applying basic boundary conditions...";

	unsigned int surfaceIndex;
	ReactionDiffusionSolverFV *fv;
	unsigned int fieldIndex;
	std::vector<std::tuple<short, std::string> > bLocSpecs = std::vector<std::tuple<short, std::string> >(2, std::tuple<unsigned int, std::string>(0, ""));

	//		z boundaries

	if (maxNeighborIndex > 3) {
		CC3D_Log(LOG_DEBUG) << "   along z-boundaries...";
		bLocSpecs[0] = tuple<short, std::string>(0, "MinZ");
		bLocSpecs[1] = tuple<short, std::string>(fieldDim.z - 1, "MaxZ");
		for (short x = 0; x < fieldDim.x; ++x)
			for (short y = 0; y < fieldDim.y; ++y)
				for (const auto& bLocSpec: bLocSpecs){
					short z = std::get<0>(bLocSpec);
					surfaceIndex = getSurfaceIndexByName(std::get<1>(bLocSpec));
					fv = getFieldFV(Point3D(x, y, z));
					for(const auto& fieldName : concentrationFieldNameVector){
						fieldIndex = getFieldIndexByName(fieldName);
						std::string bcName = std::get<0>(basicBCData[fieldIndex][surfaceIndex]);
						float bcVal = std::get<1>(basicBCData[fieldIndex][surfaceIndex]);
						if (bcName == "ConstantValue") { fv->useFixedConcentration(fieldIndex, surfaceIndex, bcVal); }
						else if (bcName == "ConstantDerivative") { fv->useFixedFluxSurface(fieldIndex, surfaceIndex, bcVal); }
					}
				}
	}

	//		y boundaries

	CC3D_Log(LOG_DEBUG) <<  "   along y-boundaries...";

	bLocSpecs[0] = tuple<short, std::string>(0, "MinY");
	bLocSpecs[1] = tuple<short, std::string>(fieldDim.y - 1, "MaxY");
	for (short x = 0; x < fieldDim.x; ++x)
		for (short z = 0; z < fieldDim.z; ++z)
			for(const auto& bLocSpec : bLocSpecs){
				short y = std::get<0>(bLocSpec);
				surfaceIndex = getSurfaceIndexByName(std::get<1>(bLocSpec));
				fv = getFieldFV(Point3D(x, y, z));
				for(const auto& fieldName : concentrationFieldNameVector){
					fieldIndex = getFieldIndexByName(fieldName);
					std::string bcName = std::get<0>(basicBCData[fieldIndex][surfaceIndex]);
					float bcVal = std::get<1>(basicBCData[fieldIndex][surfaceIndex]);
					if (bcName == "ConstantValue") { fv->useFixedConcentration(fieldIndex, surfaceIndex, bcVal); }
					else if (bcName == "ConstantDerivative") { fv->useFixedFluxSurface(fieldIndex, surfaceIndex, bcVal); }
				}
			}

	//		x boundaries

	CC3D_Log(LOG_DEBUG) << "   along x-boundaries...";

	bLocSpecs[0] = tuple<short, std::string>(0, "MinX");
	bLocSpecs[1] = tuple<short, std::string>(fieldDim.x - 1, "MaxX");
	for (short y = 0; y < fieldDim.y; ++y)
		for (short z = 0; z < fieldDim.z; ++z)
			for(const auto& bLocSpec : bLocSpecs){
				short x = std::get<0>(bLocSpec);
				surfaceIndex = getSurfaceIndexByName(std::get<1>(bLocSpec));
				fv = getFieldFV(Point3D(x, y, z));
				for(const auto& fieldName : concentrationFieldNameVector){
					fieldIndex = getFieldIndexByName(fieldName);
					std::string bcName = std::get<0>(basicBCData[fieldIndex][surfaceIndex]);
					float bcVal = std::get<1>(basicBCData[fieldIndex][surfaceIndex]);
					if (bcName == "ConstantValue") { fv->useFixedConcentration(fieldIndex, surfaceIndex, bcVal); }
					else if (bcName == "ConstantDerivative") { fv->useFixedFluxSurface(fieldIndex, surfaceIndex, bcVal); }
				}
			}

}

void ReactionDiffusionSolverFVM::initializeFieldUsingEquation(unsigned int _fieldIndex, std::string _expr) { // Derived from DiffusableVectorCommon::initializeFieldUsingEquation
	Point3D pt;
	mu::Parser parser;
	double xVar, yVar, zVar; //variables used by parser
	try {
		parser.DefineVar("x", &xVar);
		parser.DefineVar("y", &yVar);
		parser.DefineVar("z", &zVar);
		parser.SetExpr(_expr);

		for (int x = 0; x < fieldDim.x; ++x)
			for (int y = 0; y < fieldDim.y; ++y)
				for (int z = 0; z < fieldDim.z; ++z) {
					pt.x = x;
					pt.y = y;
					pt.z = z;
					//setting parser variables
					xVar = x;
					yVar = y;
					zVar = z;
					getFieldFV(pt)->setConcentration(_fieldIndex, static_cast<double>(parser.Eval()));
				}

	}
	catch (mu::Parser::exception_type &e) {
		CC3D_Log(LOG_DEBUG) << e.GetMsg();
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}


////////////////////////////////////////////////////////////// FV interface //////////////////////////////////////////////////////////////

Point3D ReactionDiffusionSolverFVM::getCoordsOfFV(ReactionDiffusionSolverFV *_fv) { return _fv->getCoords(); }

CellG * ReactionDiffusionSolverFVM::FVtoCellMap(ReactionDiffusionSolverFV * _fv) { return potts->getCellFieldG()->get(_fv->getCoords()); }

void ReactionDiffusionSolverFVM::useConstantDiffusivity(unsigned int _fieldIndex, double _diffusivityCoefficient) {	
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		_fieldFVs[i]->useConstantDiffusivity(_fieldIndex, _diffusivityCoefficient);
	}
}
void ReactionDiffusionSolverFVM::useConstantDiffusivityByType(unsigned int _fieldIndex, double _diffusivityCoefficient) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		_fieldFVs[i]->useConstantDiffusivityById(_fieldIndex, _diffusivityCoefficient);
	}
}
void ReactionDiffusionSolverFVM::useFieldDiffusivityInMedium(unsigned int _fieldIndex) {
	initDiffusivityField(_fieldIndex);
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		_fieldFVs[i]->useFieldDiffusivityInMedium(_fieldIndex);
	}
}
void ReactionDiffusionSolverFVM::useFieldDiffusivityEverywhere(unsigned int _fieldIndex) {
	initDiffusivityField(_fieldIndex);
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		_fieldFVs[i]->useFieldDiffusivityEverywhere(_fieldIndex);
	}
}
void ReactionDiffusionSolverFVM::initDiffusivityField(unsigned int _fieldIndex) {
	if (!diffusivityFieldInitialized[_fieldIndex]) {
		diffusivityFieldIndexToFieldMap[_fieldIndex] = new WatchableField3D<float>(fieldDim, 0.0);
		sim->registerConcentrationField(concentrationFieldNameVector[_fieldIndex] + diffusivityFieldSuffixStd, diffusivityFieldIndexToFieldMap[_fieldIndex]);
		diffusivityFieldInitialized[_fieldIndex] = true;
	}
}

void ReactionDiffusionSolverFVM::setDiffusivityFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val) { diffusivityFieldIndexToFieldMap[_fieldIndex]->set(_pt, _val); }

float ReactionDiffusionSolverFVM::getConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt) { return getFieldFV(_pt)->getConcentration(_fieldIndex); }

void ReactionDiffusionSolverFVM::setConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val) { getFieldFV(_pt)->setConcentration(_fieldIndex, _val); }

void ReactionDiffusionSolverFVM::useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _outwardFluxVal, ReactionDiffusionSolverFV *_fv) { 
	_fv->useFixedFluxSurface(_fieldIndex, _surfaceIndex, _outwardFluxVal);
}

void ReactionDiffusionSolverFVM::useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _val, ReactionDiffusionSolverFV *_fv) { 
	_fv->useFixedConcentration(_fieldIndex, _surfaceIndex, _val);
}

void ReactionDiffusionSolverFVM::useFixedFVConcentration(unsigned int _fieldIndex, float _val, ReactionDiffusionSolverFV *_fv) { 
	_fv->useFixedFVConcentration(_fieldIndex, _val);
}

void ReactionDiffusionSolverFVM::useDiffusiveSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, Point3D _pt) {
	fieldFVs[pt2ind(_pt)]->useDiffusiveSurface(_fieldIndex, _surfaceIndex);
}

void ReactionDiffusionSolverFVM::useDiffusiveSurfaces(unsigned int _fieldIndex, Point3D _pt) {
	fieldFVs[pt2ind(_pt)]->useDiffusiveSurfaces(_fieldIndex);
}

void ReactionDiffusionSolverFVM::useDiffusiveSurfaces(unsigned int _fieldIndex) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		_fieldFVs[i]->useDiffusiveSurfaces(_fieldIndex);
	}
}

void ReactionDiffusionSolverFVM::usePermeableSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, Point3D _pt) {
	fieldFVs[pt2ind(_pt)]->usePermeableSurface(_fieldIndex, _surfaceIndex);
}

void ReactionDiffusionSolverFVM::usePermeableSurfaces(unsigned int _fieldIndex, bool _activate) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (_fieldFVs)
	for (int i=0;i<_fieldFVs.size();i++){
		_fieldFVs[i]->usePermeableSurfaces(_fieldIndex, _activate);
	}
}

void ReactionDiffusionSolverFVM::updateSurfaceAreas() {
	// To be updated for hex lattices
	surfaceAreas = std::vector<float>(maxNeighborIndex, 0.0);
	surfaceAreas[0] = lengthY * lengthZ;
	surfaceAreas[1] = lengthX * lengthZ;
	surfaceAreas[2] = lengthY * lengthZ;
	surfaceAreas[3] = lengthX * lengthZ;
	if (maxNeighborIndex > 3) {
		surfaceAreas[4] = lengthX * lengthY;
		surfaceAreas[5] = lengthX * lengthY;
	}
}

///////////////////////////////////////////////////////////// Cell interface /////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFVM::initializeCellData(const CellG *_cell, unsigned int _numFields) {
	ReactionDiffusionSolverFVMCellData *cellData = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr);
	cellData->permeationCoefficients = std::vector<std::vector<double> >(_numFields, std::vector<double>(numCellTypes, 0.0));
	cellData->permeableBiasCoefficients = std::vector<std::vector<double> >(_numFields, std::vector<double>(numCellTypes, 1.0));
	cellData->diffusivityCoefficients = std::vector<double>(_numFields, 0.0);
	cellData->outwardFluxValues = std::vector<double>(_numFields, 0.0);
}

void ReactionDiffusionSolverFVM::initializeCellData(unsigned int _numFields) {
	if (!(pixelTrackerPlugin->fullyInitialized())) { pixelTrackerPlugin->fullTrackerDataInit(); }

	CellInventory *cellInventory = &potts->getCellInventory();

	for (CellInventory::cellInventoryIterator cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr)
		initializeCellData(cellInventory->getCell(cell_itr), _numFields);
}

double ReactionDiffusionSolverFVM::getCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex) {

	ASSERT_OR_THROW("Inappropriate call to medium coefficient.", _cell);

	ReactionDiffusionSolverFVMCellData *cellData = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr);
	std::vector<double> &diffusivityCoefficients = cellData->diffusivityCoefficients;
	if (diffusivityCoefficients.size() < _fieldIndex) { 

		CC3D_Log(LOG_DEBUG) << "Initializing cell diffusivity coefficient on the fly...";

		cellData->diffusivityCoefficients = std::vector<double>(numFields);
		for (unsigned int i = 0; i < numFields; ++i) { setCellDiffusivityCoefficient(_cell, _fieldIndex); }
		std::vector<double> &diffusivityCoefficients = cellData->diffusivityCoefficients;
	}
	return diffusivityCoefficients[_fieldIndex];
}

void ReactionDiffusionSolverFVM::setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex, double _diffusivityCoefficient) {
	std::vector<double> &diffusivityCoefficients = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr)->diffusivityCoefficients;
	diffusivityCoefficients[_fieldIndex] = _diffusivityCoefficient;
}

void ReactionDiffusionSolverFVM::setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex) {
	setCellDiffusivityCoefficient(_cell, _fieldIndex, constantDiffusionCoefficientsVecCellType[_fieldIndex][(int)(_cell->type)]);}

void ReactionDiffusionSolverFVM::setCellDiffusivityCoefficients(unsigned int _fieldIndex) {
	CellInventory *cellInventory = &potts->getCellInventory();

	for (CellInventory::cellInventoryIterator cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr) {
		setCellDiffusivityCoefficient(cellInventory->getCell(cell_itr), _fieldIndex);
	}
}

void ReactionDiffusionSolverFVM::setCellDiffusivityCoefficients() {
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { setCellDiffusivityCoefficients(fieldIndex); }
}

std::vector<double> ReactionDiffusionSolverFVM::getPermeableCoefficients(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex) {

	ReactionDiffusionSolverFVMCellData *cellData = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr);

	double permeationCoeffecient = cellData->permeationCoefficients[_fieldIndex][_nCellTypeId];
	double biasCoeff = cellData->permeableBiasCoefficients[_fieldIndex][_nCellTypeId];

	return std::vector<double>{ permeationCoeffecient, biasCoeff };

}

std::vector<double> ReactionDiffusionSolverFVM::getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, unsigned int _fieldIndex) {

	unsigned int nTypeInt;
	double permeationCoeffecient, biasCoeff, nBiasCoeff;

	if (_cell) { 
		ReactionDiffusionSolverFVMCellData *cellData = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr);

		if (_nCell) { 
			nTypeInt = (unsigned int)(_nCell->type);
			nBiasCoeff = ReactionDiffusionSolverFVMCellDataAccessor.get(_nCell->extraAttribPtr)->permeableBiasCoefficients[_fieldIndex][(unsigned int)_cell->type];
		}
		else { 
			nTypeInt = 0;
			nBiasCoeff = constPermBiasCoeffsVecCellType[_fieldIndex][nTypeInt][(unsigned int)_cell->type];
		}

		permeationCoeffecient = cellData->permeationCoefficients[_fieldIndex][nTypeInt];
		biasCoeff = cellData->permeableBiasCoefficients[_fieldIndex][nTypeInt];
	}
	else { 

		if (_nCell) { 
			nTypeInt = (unsigned int)(_nCell->type);
			nBiasCoeff = ReactionDiffusionSolverFVMCellDataAccessor.get(_nCell->extraAttribPtr)->permeableBiasCoefficients[_fieldIndex][0];
		}
		else {
			nTypeInt = 0;
			nBiasCoeff = constPermBiasCoeffsVecCellType[_fieldIndex][nTypeInt][0];
		}

		permeationCoeffecient = constantPermeationCoefficientsVecCellType[_fieldIndex][0][nTypeInt];
		biasCoeff = constPermBiasCoeffsVecCellType[_fieldIndex][0][nTypeInt];
	}

	return std::vector<double>{ permeationCoeffecient, biasCoeff, nBiasCoeff };
}

void ReactionDiffusionSolverFVM::setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeationCoefficient) {
	std::vector<std::vector<double> > &permeationCoefficients = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr)->permeationCoefficients;
	permeationCoefficients[_fieldIndex][_nCellTypeId] = _permeationCoefficient;
}

void ReactionDiffusionSolverFVM::setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex) {
	setCellPermeationCoefficient(_cell, _nCellTypeId, _fieldIndex, constantPermeationCoefficientsVecCellType[_fieldIndex][(int)(_cell->type)][_nCellTypeId]);}

void ReactionDiffusionSolverFVM::setCellPermeationCoefficients(unsigned int _fieldIndex) {
	CellInventory *cellInventory = &potts->getCellInventory();

	for (CellInventory::cellInventoryIterator cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr)
		for (unsigned int nCellTypeId = 0; nCellTypeId < numCellTypes; ++nCellTypeId)
			setCellPermeationCoefficient(cellInventory->getCell(cell_itr), nCellTypeId, _fieldIndex);
}

void ReactionDiffusionSolverFVM::setCellPermeationCoefficients() {
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { setCellPermeationCoefficients(fieldIndex); }
}

void ReactionDiffusionSolverFVM::setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeableBiasCoefficient) {
	std::vector<std::vector<double> > &permeableBiasCoefficients = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr)->permeableBiasCoefficients;
	permeableBiasCoefficients[_fieldIndex][_nCellTypeId] = _permeableBiasCoefficient;
}

void ReactionDiffusionSolverFVM::setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex) {
	setCellPermeableBiasCoefficient(_cell, _nCellTypeId, _fieldIndex, constPermBiasCoeffsVecCellType[_fieldIndex][(int)(_cell->type)][_nCellTypeId]);
}

void ReactionDiffusionSolverFVM::setCellPermeableBiasCoefficients(unsigned int _fieldIndex) {
	CellInventory *cellInventory = &potts->getCellInventory();

	for (CellInventory::cellInventoryIterator cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr)
		for (unsigned int nCellTypeId = 0; nCellTypeId < numCellTypes; ++nCellTypeId)
			setCellPermeableBiasCoefficient(cellInventory->getCell(cell_itr), nCellTypeId, _fieldIndex);
}

void ReactionDiffusionSolverFVM::setCellPermeableBiasCoefficients() {
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { setCellPermeableBiasCoefficients(fieldIndex); }
}

void ReactionDiffusionSolverFVM::setCellPermeableCoefficients() {
	setCellPermeationCoefficients();
	setCellPermeableBiasCoefficients();
}

// Not currently implemented.
double ReactionDiffusionSolverFVM::getCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex) {
	std::vector<double> &outwardFluxValues = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr)->outwardFluxValues;
	return outwardFluxValues[_fieldIndex];
}

// Not currently implemented.
void ReactionDiffusionSolverFVM::setCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex, float _outwardFlux) {
	std::vector<double> &outwardFluxValues = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr)->outwardFluxValues;
	outwardFluxValues[_fieldIndex] = _outwardFlux;

	bool hasOutwardFlux = _outwardFlux != 0;
	if(!hasOutwardFlux) {
		CellInventory *cellInventory = &potts->getCellInventory();
		for(CellInventory::cellInventoryIterator cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr) {
			for(unsigned int i = 0; i < numFields; i++) {
				hasOutwardFlux |= getCellOutwardFlux(cell_itr->second, i) != 0;
				if(hasOutwardFlux)
					break;
			}
			if(hasOutwardFlux)
				break;
		}
	}

	auto _fieldFVs = fieldFVs;
	#pragma omp parallel for shared(_fieldFVs, hasOutwardFlux)
	for(int i = 0; i < _fieldFVs.size(); i++) {
		_fieldFVs[i]->useCellInterfaceFlux(hasOutwardFlux);
	}
}

//////////////////////////////////////////////////////////// Solver functions ////////////////////////////////////////////////////////////

bool ReactionDiffusionSolverFVM::inContact(CellG *cell, const unsigned char &typeIndex) {
	NeighborTracker *nt = neighborTrackerPlugin->getNeighborTrackerAccessorPtr()->get(cell->extraAttribPtr);
	for(auto &ntd : nt->cellNeighbors) {
		const unsigned char nTypeIndex = ntd.neighborAddress ? ntd.neighborAddress->type : 0;
		if(typeIndex == nTypeIndex)
			return true;
	}
	return false;
}

std::set<unsigned char> ReactionDiffusionSolverFVM::getNeighbors(CellG *cell) {
	std::set<unsigned char> result;

	NeighborTracker *nt = neighborTrackerPlugin->getNeighborTrackerAccessorPtr()->get(cell->extraAttribPtr);
	for(auto &ntd : nt->cellNeighbors)
		result.insert(ntd.neighborAddress ? ntd.neighborAddress->type : 0);

	return result;
}

unsigned int ReactionDiffusionSolverFVM::getSurfaceIndexByName(std::string _surfaceName) {
	std::map<std::string, unsigned int>::iterator m_itr = surfaceMapNameToIndex.find(_surfaceName);
	if (m_itr != surfaceMapNameToIndex.end()) { return m_itr->second; }
	else { ASSERT_OR_THROW(string("Unknown surface name: ") + _surfaceName, false); }
}

unsigned int ReactionDiffusionSolverFVM::getFieldIndexByName(std::string _fieldName) {
	std::map<std::string, unsigned int>::iterator m_itr = fieldNameToIndexMap.find(_fieldName);
	if (m_itr != fieldNameToIndexMap.end()) { return m_itr->second; }
	else { ASSERT_OR_THROW(string("Unknown field name: ") + _fieldName, false); }
}

void ReactionDiffusionSolverFVM::setUnitsTime(std::string _unitsTime) {
	std::vector<std::string>::iterator mitr = find(availableUnitsTime.begin(), availableUnitsTime.end(), _unitsTime);
	if (mitr != availableUnitsTime.end()) {
		for (int i = 0; i < availableUnitsTime.size(); ++i) {
			if (availableUnitsTime[i] == _unitsTime) {
				unitsTime = _unitsTime;
				unitTimeConv = availableUnitTimeConv[i];
				incTime *= unitTimeConv;
				return;
			}
		}
	}
	ASSERT_OR_THROW(string("Unrecognized time unit: ") + _unitsTime, false);
}

void ReactionDiffusionSolverFVM::setLengths(float _lengthX, float _lengthY, float _lengthZ) {
	lengthX = _lengthX;
	lengthY = _lengthY;
	lengthZ = _lengthZ;
	signedDistanceBySurfaceIndex = std::vector<float>(maxNeighborIndex, 0.0);
	for (int surfaceIndex = 0; surfaceIndex < surfaceNormSign.size(); ++surfaceIndex) {
		signedDistanceBySurfaceIndex[surfaceIndex] = ((float)surfaceNormSign[surfaceIndex]) * getLength(indexMapSurfToCoord[surfaceIndex]);
	}
	updateSurfaceAreas();
}

std::map<unsigned int, ReactionDiffusionSolverFV *> ReactionDiffusionSolverFVM::getFVNeighborFVs(ReactionDiffusionSolverFV *_fv) {
	maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
	Neighbor neighbor;
	std::map<unsigned int, ReactionDiffusionSolverFV *> neighborFVs;

	Point3D pt = _fv->getCoords();

	for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
		neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
		if (neighbor.distance) { neighborFVs.insert(make_pair(nIdx, getFieldFV(neighbor.pt))); }
		else { neighborFVs.insert(make_pair(nIdx, nullptr)); }
	}

	return neighborFVs;
}

Point3D ReactionDiffusionSolverFVM::ind2pt(unsigned int _ind) {
	Point3D pt;
	pt.x = _ind % fieldDim.x;
	pt.y = (_ind - pt.x) / fieldDim.x % fieldDim.y;
	pt.z = ((_ind - pt.x) / fieldDim.x - pt.y) / fieldDim.y;
	return pt;
}

unsigned int ReactionDiffusionSolverFVM::pt2ind(const Point3D &_pt, Dim3D _fieldDim) {
	int ind = _pt.x + _fieldDim.x * (_pt.y + _fieldDim.y * _pt.z);
	if (ind < 0 || ind > fieldDim.x*fieldDim.y*fieldDim.z) { throw CC3DException("Point is not valid."); }
	return (unsigned int)(ind);
}

///////////////////////////////////////////////////////////// Solver interface ///////////////////////////////////////////////////////////

void ReactionDiffusionSolverFVM::setFieldExpressionMultiplier(unsigned int _fieldIndex, std::string _expr) { fieldExpressionStringsMergedDiag[_fieldIndex] = _expr; }

void ReactionDiffusionSolverFVM::setFieldExpressionIndependent(unsigned int _fieldIndex, std::string _expr) { fieldExpressionStringsMergedOffDiag[_fieldIndex] = _expr; }

/////////////////////////////////////////////////////////////// Field wrap ///////////////////////////////////////////////////////////////

template <class T>
void RDFVMField3DWrap<T>::set(const Point3D &pt, const T value) { solver->setConcentrationFieldPtVal(fieldName, pt, value); }

template <class T>
T RDFVMField3DWrap<T>::get(const Point3D &pt) const { return solver->getConcentrationFieldPtVal(fieldName, pt); }

template <class T>
void RDFVMField3DWrap<T>::setByIndex(long _offset, const T value) { solver->setConcentrationFieldPtVal(fieldName, solver->ind2pt(_offset), value); }

template <class T>
T RDFVMField3DWrap<T>::getByIndex(long _offset) const { return solver->getConcentrationFieldPtVal(fieldName, solver->ind2pt(_offset)); }

template <class T>
Dim3D RDFVMField3DWrap<T>::getDim() const { return solver->getFieldDim(); }

template <class T>
bool RDFVMField3DWrap<T>::isValid(const Point3D &pt) const { return solver->isValid(pt); }

////////////////////////////////////////////////////////////// Finite volume /////////////////////////////////////////////////////////////

ReactionDiffusionSolverFV::ReactionDiffusionSolverFV(ReactionDiffusionSolverFVM *_solver, Point3D _coords, int _numFields) :
	usingCellInterfaceFlux{false},
	solver(_solver),
	coords(_coords)
{
	concentrationVecAux = std::vector<double>(_numFields, 0.0);
	concentrationVecOld = std::vector<double>(_numFields, 0.0);
	secrRateStorage = std::vector<double>(_numFields, 0.0);
	auxVars = std::vector<std::vector<double> >(_numFields, std::vector<double>(1, 0.0));
	physTime = 0.0;
	stabilityMode = false;

	diagonalFunctions = std::vector<mu::Parser>(_numFields, templateParserFunction());
	offDiagonalFunctions = std::vector<mu::Parser>(_numFields, templateParserFunction());

	fieldDiffusivityFunctionPtrs = std::vector<DiffusivityFunction>(_numFields, &ReactionDiffusionSolverFV::returnZero);
}

void ReactionDiffusionSolverFV::initialize() {
	neighborFVs = solver->getFVNeighborFVs(this);
	bcVals = std::vector<std::vector<double> >(
		concentrationVecOld.size(), 
		std::vector<double>(neighborFVs.size(), 0.0));
	surfaceFluxFunctionPtrs = std::vector<std::vector<FluxFunction> >(
		concentrationVecOld.size(), 
		std::vector<FluxFunction>(neighborFVs.size(), &ReactionDiffusionSolverFV::returnZero));
}

void ReactionDiffusionSolverFV::secrete(const std::vector<SecretionData> &secrFieldVec) {
	CellG *cell = solver->FVtoCellMap(this);
	const unsigned char cellTypeId = cell ? cell->type : 0;

	for(unsigned int i = 0; i < secrRateStorage.size(); i++) {
		SecretionData secrData = secrFieldVec[i];
		double secrRate = 0.0;

		if(secrData.secretionTypeIds.find(cellTypeId) != secrData.secretionTypeIds.end())
			secrRate = secreteSingleField(i, cellTypeId, secrData);
		else if(secrData.secretionOnContactTypeIds.find(cellTypeId) != secrData.secretionOnContactTypeIds.end())
			secrRate = secreteOnContactSingleField(i, cellTypeId, secrData);
		else if(secrData.constantConcentrationTypeIds.find(cellTypeId) != secrData.constantConcentrationTypeIds.end())
			secrRate = secreteConstantConcentrationSingleField(i, cellTypeId, secrData);

		secrRateStorage[i] += secrRate;
	}
}

void ReactionDiffusionSolverFV::solve() {

	CellG *cell;
	std::vector<CellG*> nbsCells;
	if(usingCellInterfaceFlux) {
		cell = solver->FVtoCellMap(this);
		nbsCells = std::vector<CellG*>(neighborFVs.size(), 0);
		for(auto &itr : neighborFVs)
			if(itr.second) {
				while(nbsCells.size() <= itr.first) nbsCells.push_back(0);
				nbsCells[itr.first] = solver->FVtoCellMap(itr.second);
			}
	}

	std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr;
	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) {
		for (fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
			std::vector<double> fluxVals = (this->*surfaceFluxFunctionPtrs[i][fv_itr->first])(i, fv_itr->first, fv_itr->second);
			concentrationVecAux[i] += fluxVals[0] * concentrationVecOld[i] + fluxVals[2];
			if (fv_itr->second != nullptr) {
				concentrationVecAux[i] += fluxVals[1] * fv_itr->second->getConcentrationOld(i);
				if(usingCellInterfaceFlux)
					concentrationVecAux[i] += cellInterfaceFlux(i, fv_itr->first, cell, nbsCells[fv_itr->first]);
			}
		}

		concentrationVecAux[i] += diagonalFunctionEval(i) * concentrationVecOld[i] + offDiagonalFunctionEval(i);
	}
}

double ReactionDiffusionSolverFV::solveStable() {

	CellG *cell;
	std::vector<CellG*> nbsCells;
	if(usingCellInterfaceFlux) {
		cell = solver->FVtoCellMap(this);
		nbsCells = std::vector<CellG*>(neighborFVs.size(), 0);
		for(auto &itr : neighborFVs)
			if(itr.second) {
				while(nbsCells.size() <= itr.first) nbsCells.push_back(0);
				nbsCells[itr.first] = solver->FVtoCellMap(itr.second);
			}
	}

	double incTime = numeric_limits<double>::max();

	std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr;

	double den;

	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) {
		double den1 = 0.0;
		double den2 = 0.0;

		for (fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
			std::vector<double> fluxVals = (this->*surfaceFluxFunctionPtrs[i][fv_itr->first])(i, fv_itr->first, fv_itr->second);

			den1 += fluxVals[0];
			concentrationVecAux[i] += fluxVals[2];

			if (fv_itr->second != nullptr) { 
				den2 += abs(fluxVals[1]);
				concentrationVecAux[i] += fluxVals[1] * fv_itr->second->getConcentrationOld(i);
				if(usingCellInterfaceFlux)
					concentrationVecAux[i] += cellInterfaceFlux(i, fv_itr->first, fv_itr->second);
			}
		}

		den1 += diagonalFunctionEval(i);
		den = abs(den1) + 1.001 * den2;
		concentrationVecAux[i] += den1 * concentrationVecOld[i] + offDiagonalFunctionEval(i);

		if (den > 0) { incTime = min(incTime, (double) 1.0 / den); }

	}

	return incTime;
}

void ReactionDiffusionSolverFV::update(double _incTime) {
	for (unsigned int fieldIndex = 0; fieldIndex < concentrationVecOld.size(); ++fieldIndex) {
		concentrationVecOld[fieldIndex] = max(concentrationVecOld[fieldIndex] + (concentrationVecAux[fieldIndex] + secrRateStorage[fieldIndex]) * _incTime, 0.0);
		concentrationVecAux[fieldIndex] = 0.0;
		secrRateStorage[fieldIndex] = 0.0;
	}
	physTime += _incTime;
}

void ReactionDiffusionSolverFV::useConstantDiffusivity(unsigned int _fieldIndex) { useConstantDiffusivity(_fieldIndex, solver->getConstantFieldDiffusivity(_fieldIndex)); }

void ReactionDiffusionSolverFV::useConstantDiffusivity(unsigned int _fieldIndex, double _constantDiff) {
	auxVars[_fieldIndex][0] = _constantDiff;
	fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getConstantDiffusivity;
}

void ReactionDiffusionSolverFV::useConstantDiffusivityById(unsigned int _fieldIndex) { useConstantDiffusivityById(_fieldIndex, solver->getConstantFieldDiffusivity(_fieldIndex)); }

void ReactionDiffusionSolverFV::useConstantDiffusivityById(unsigned int _fieldIndex, double _constantDiff) {
	auxVars[_fieldIndex][0] = _constantDiff;
	fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getConstantDiffusivityById;
}

void ReactionDiffusionSolverFV::useDiffusiveSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex) {
	surfaceFluxFunctionPtrs[_fieldIndex][_surfaceIndex] = &ReactionDiffusionSolverFV::diffusiveSurfaceFlux;
}

void ReactionDiffusionSolverFV::useDiffusiveSurfaces(unsigned int _fieldIndex) {
	for (std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
		surfaceFluxFunctionPtrs[_fieldIndex][fv_itr->first] = &ReactionDiffusionSolverFV::diffusiveSurfaceFlux;
	}
}

void ReactionDiffusionSolverFV::usePermeableSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, bool _activate) {
	if(_activate) { surfaceFluxFunctionPtrs[_fieldIndex][_surfaceIndex] = &ReactionDiffusionSolverFV::permeableSurfaceFlux; }
	else { useDiffusiveSurface(_fieldIndex, _surfaceIndex); }
}

void ReactionDiffusionSolverFV::usePermeableSurfaces(unsigned int _fieldIndex, bool _activate) {
	if (_activate) {
		for (std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
			surfaceFluxFunctionPtrs[_fieldIndex][fv_itr->first] = &ReactionDiffusionSolverFV::permeableSurfaceFlux;
		}
	}
	else { useDiffusiveSurfaces(_fieldIndex); }
}

void ReactionDiffusionSolverFV::useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, double _outwardFluxVal) {
	bcVals[_fieldIndex][_surfaceIndex] = _outwardFluxVal;
	surfaceFluxFunctionPtrs[_fieldIndex][_surfaceIndex] = &ReactionDiffusionSolverFV::fixedSurfaceFlux;
}

void ReactionDiffusionSolverFV::useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, double _val) {
	bcVals[_fieldIndex][_surfaceIndex] = _val;
	surfaceFluxFunctionPtrs[_fieldIndex][_surfaceIndex] = &ReactionDiffusionSolverFV::fixedConcentrationFlux;
}

void ReactionDiffusionSolverFV::useFixedFVConcentration(unsigned int _fieldIndex, double _val) {
	bcVals[_fieldIndex].clear();
	bcVals[_fieldIndex] = std::vector<double>(1, _val);
	for (unsigned int _surfaceIndex = 0; _surfaceIndex < surfaceFluxFunctionPtrs[_fieldIndex].size(); ++_surfaceIndex) {
		surfaceFluxFunctionPtrs[_fieldIndex][_surfaceIndex] = &ReactionDiffusionSolverFV::fixedFVConcentrationFlux;
		diagonalFunctions[_fieldIndex] = zeroMuParserFunction();
		offDiagonalFunctions[_fieldIndex] = zeroMuParserFunction();
	}
}

void ReactionDiffusionSolverFV::registerFieldSymbol(unsigned int _fieldIndex, std::string _fieldSymbol) {
	for (unsigned int parserIndex = 0; parserIndex < diagonalFunctions.size(); ++parserIndex) {
		diagonalFunctions[parserIndex].DefineVar(_fieldSymbol, &concentrationVecOld[_fieldIndex]);
		offDiagonalFunctions[parserIndex].DefineVar(_fieldSymbol, &concentrationVecOld[_fieldIndex]);
	}
}

void ReactionDiffusionSolverFV::setDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr) { 
	try { diagonalFunctions[_fieldIndex].SetExpr(_expr); }
	catch (mu::Parser::exception_type &e) {
		CC3D_Log(LOG_DEBUG) << e.GetMsg();
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}

void ReactionDiffusionSolverFV::setOffDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr) { 
	try { offDiagonalFunctions[_fieldIndex].SetExpr(_expr); }
	catch (mu::Parser::exception_type &e) {
		CC3D_Log(LOG_DEBUG) << e.GetMsg();
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}

mu::Parser ReactionDiffusionSolverFV::zeroMuParserFunction() {
	mu::Parser _zeroFcn = mu::Parser();
	_zeroFcn.SetExpr("0.0");
	return _zeroFcn;
}

mu::Parser ReactionDiffusionSolverFV::templateParserFunction() {
	mu::Parser _fcn = zeroMuParserFunction();

	_fcn.DefineConst("x", (double)coords.x);
	_fcn.DefineConst("y", (double)coords.y);
	_fcn.DefineConst("z", (double)coords.z);
	_fcn.DefineVar("t", &physTime);
	return _fcn;
}

void ReactionDiffusionSolverFV::setConcentrationVec(std::vector<double> _concentrationVec) {
	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) { concentrationVecOld[i] = _concentrationVec[i]; }
}

void ReactionDiffusionSolverFV::addConcentrationVecIncrement(std::vector<double> _concentrationVecInc) {
	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) { 
		concentrationVecOld[i] += _concentrationVecInc[i];
		concentrationVecOld[i] = max(concentrationVecOld[i], 0.0);
	}
}

double ReactionDiffusionSolverFV::diagonalFunctionEval(unsigned int _fieldIndex) { return diagonalFunctions[_fieldIndex].Eval(); }

double ReactionDiffusionSolverFV::offDiagonalFunctionEval(unsigned int _fieldIndex) { return offDiagonalFunctions[_fieldIndex].Eval(); }

double ReactionDiffusionSolverFV::getConstantDiffusivityById(unsigned int _fieldIndex) {
	CellG * cell = solver->FVtoCellMap(this);
	if (cell == 0) { return auxVars[_fieldIndex][0]; }
	else { return solver->getCellDiffusivityCoefficient(cell, _fieldIndex); }
}

double ReactionDiffusionSolverFV::getFieldDiffusivityField(unsigned int _fieldIndex) { return solver->getDiffusivityFieldPtVal(_fieldIndex, coords); }

double ReactionDiffusionSolverFV::getFieldDiffusivityInMedium(unsigned int _fieldIndex) {
	CellG * cell = solver->FVtoCellMap(this);
	if (cell == 0) { return getFieldDiffusivityField(_fieldIndex); }
	else { return solver->getCellDiffusivityCoefficient(cell, _fieldIndex); }
}

double ReactionDiffusionSolverFV::cellInterfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, CellG *cell, CellG *nCell) {
	double result = 0.0;

	if(!cell || nCell == cell) return result;

	double outwardFluxVal = solver->getCellOutwardFlux(cell, _fieldIndex);
	if(nCell) outwardFluxVal -= solver->getCellOutwardFlux(nCell, _fieldIndex);
	double diffC = getFieldDiffusivity(_fieldIndex);
	double length = (double)(solver->getLengthBySurfaceIndex(_surfaceIndex));
	return -diffC * outwardFluxVal / length;
}

double ReactionDiffusionSolverFV::cellInterfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	CellG *cell = solver->FVtoCellMap(this);
	if(!cell) return 0.0;

	CellG *nCell = solver->FVtoCellMap(_nFv);
	if(nCell == cell) return 0.0;

	return cellInterfaceFlux(_fieldIndex, _surfaceIndex, cell, nCell);
}

std::vector<double> ReactionDiffusionSolverFV::diffusiveSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	if (_nFv == nullptr) {
		CC3D_Log(LOG_DEBUG) << "Warning: diffusive surface flux for an unconnected FV pair!" ;
		return std::vector<double>{0.0, 0.0, 0.0};
	}

	double diffC = getFieldDiffusivity(_fieldIndex);
	double nDiffC = _nFv->getFieldDiffusivity(_fieldIndex);
	double surfaceDiffC;
	if (diffC * nDiffC == 0) { return std::vector<double>{0.0, 0.0, 0.0}; }
	else { surfaceDiffC = 2 * diffC * nDiffC / (diffC + nDiffC); }
	double length = solver->getLengthBySurfaceIndex(_surfaceIndex);
	double outVal = surfaceDiffC / length / length;
	return std::vector<double>{-outVal, outVal, 0.0};
}

std::vector<double> ReactionDiffusionSolverFV::permeableSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	if (_nFv == nullptr) {
		CC3D_Log(LOG_DEBUG) << "Warning: permeable surface flux for an unconnected FV pair!";
		return std::vector<double>{0.0, 0.0, 0.0};
	}
	
	CellG *cell = solver->FVtoCellMap(this);
	CellG *nCell = solver->FVtoCellMap(_nFv);

	if (cell == nCell) { return diffusiveSurfaceFlux(_fieldIndex, _surfaceIndex, _nFv); }

	std::vector<double> permeableCoefficients = solver->getPermeableCoefficients(cell, nCell, _fieldIndex);
	double length = (double)(solver->getLengthBySurfaceIndex(_surfaceIndex));
	double outVal = permeableCoefficients[0] / length;
	return std::vector<double>{-permeableCoefficients[1] * outVal, permeableCoefficients[2] * outVal, 0.0};
}

std::vector<double> ReactionDiffusionSolverFV::fixedSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	double diffC = getFieldDiffusivity(_fieldIndex);
	double outwardFluxVal = bcVals[_fieldIndex][_surfaceIndex];
	double length = (double)(solver->getLengthBySurfaceIndex(_surfaceIndex));
	return std::vector<double>{0.0, 0.0, -diffC * outwardFluxVal / length};
}

std::vector<double> ReactionDiffusionSolverFV::fixedConcentrationFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	double diffC = getFieldDiffusivity(_fieldIndex);
	double fixedVal = bcVals[_fieldIndex][_surfaceIndex];
	double length = (double)(solver->getLengthBySurfaceIndex(_surfaceIndex));
	double outVal = 2 * diffC / length / length;
	return std::vector<double>{-outVal, 0.0, outVal * fixedVal};
}

std::vector<double> ReactionDiffusionSolverFV::fixedFVConcentrationFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	concentrationVecOld[_fieldIndex] = bcVals[_fieldIndex][0];
	return std::vector<double>{0.0, 0.0, 0.0};
}

double ReactionDiffusionSolverFV::secreteSingleField(const unsigned int &fieldIndex, const unsigned char &typeIndex, const SecretionData &secrData) {
	double result = 0.0;
	auto itr_SecrConstMap = secrData.typeIdSecrConstMap.find(typeIndex);
	result += itr_SecrConstMap == secrData.typeIdSecrConstMap.end() ? 0.0 : itr_SecrConstMap->second;

	auto itr_UptakeDataMap = secrData.typeIdUptakeDataMap.find(typeIndex);
	if(itr_UptakeDataMap != secrData.typeIdUptakeDataMap.end())
		result -= std::min<double>(itr_UptakeDataMap->second.maxUptake, concentrationVecOld[fieldIndex] * itr_UptakeDataMap->second.relativeUptakeRate);

	return result;
}

double ReactionDiffusionSolverFV::secreteOnContactSingleField(const unsigned int &fieldIndex, const unsigned char &typeIndex, const SecretionData &secrData) {
	double result = 0.0;

	CellG* cell = solver->FVtoCellMap(this);
	if(!cell)
		return result;

	auto itr = secrData.typeIdSecrOnContactDataMap.find(typeIndex);
	if(itr == secrData.typeIdSecrOnContactDataMap.end())
		return result;

	auto contactCellMapPtr = itr->second.contactCellMap;
	auto neighborTypeIds = solver->getNeighbors(cell);
	for(auto &nTypeId : neighborTypeIds) {
		auto contactCellMapPtr_itr = contactCellMapPtr.find(nTypeId);
		if(contactCellMapPtr_itr != contactCellMapPtr.end())
			result += contactCellMapPtr_itr->second;;
	}

	return result;
}

double ReactionDiffusionSolverFV::secreteConstantConcentrationSingleField(const unsigned int &fieldIndex, const unsigned char &typeIndex, const SecretionData &secrData) {
	auto itr = secrData.typeIdSecrConstConstantConcentrationMap.find(typeIndex);
	if(itr != secrData.typeIdSecrConstConstantConcentrationMap.end())
		concentrationVecOld[fieldIndex] = itr->second;
	return 0.0;
}


double ReactionDiffusionSolverFV::getFieldDiffusivity(unsigned int _fieldIndex) { return (this->*fieldDiffusivityFunctionPtrs[_fieldIndex])(_fieldIndex); }

double ReactionDiffusionSolverFV::getCellOutwardFlux(unsigned int _fieldIndex) {
	CellG *cell = solver->FVtoCellMap(this);
	return cell ? solver->getCellOutwardFlux(cell, _fieldIndex) : 0.0;
}

std::vector<double> ReactionDiffusionSolverFV::getCellOutwardFluxes() {
	CellG *cell = solver->FVtoCellMap(this);

	if(!cell) return std::vector<double>(concentrationVecOld.size(), 0.0);

	std::vector<double> result;
	result.reserve(concentrationVecOld.size());
	for(unsigned int i = 0; i < concentrationVecOld.size(); i++)
		result.push_back(solver->getCellOutwardFlux(cell, i));

	return result;
}

///////////////////////////////////////////////////////////// Cell parameters/////////////////////////////////////////////////////////////
