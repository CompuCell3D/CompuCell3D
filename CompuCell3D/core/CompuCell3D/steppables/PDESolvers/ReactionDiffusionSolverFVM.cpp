#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>
#include <BasicUtils/BasicClassGroup.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <PublicUtilities/StringUtils.h>
#include <muParser/muParser.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <omp.h>
// #include <concurrent_vector.h>
// #include <concurrent_unordered_set.h>
// #include <ppl.h>

// macro to ensure CC3d_log is enabled only when debugging
#ifdef DEBUG
#define CC3d_log(x) std::cerr <<x<<std::endl
#else
#define CC3d_log(x)
#endif

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
	simpleMassConservation = false;
	usingECMaterials = false;
	cellDataLoaded = false;
	integrationTimeStep = incTime;

	physTime = 0.0;
	flipSourcePt = Point3D();

	field3DAdditionalPtFcnPtr = &ReactionDiffusionSolverFVM::field3DAdditionalPtPreStartup;
	field3DChangeFcnPtr = &ReactionDiffusionSolverFVM::field3DChangePreStartup;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFVM::~ReactionDiffusionSolverFVM()
{
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr = 0;

	delete fvMaxStableTimeSteps;
	fvMaxStableTimeSteps = 0;

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { 
		delete concentrationFieldVector[fieldIndex];
		concentrationFieldVector[fieldIndex] = 0;
	}
}

void ReactionDiffusionSolverFVM::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


	CC3d_log("*******************************");
	CC3d_log("* Begin RDFVM initialization! *");
	CC3d_log("*******************************");
	// cerr << "*******************************" << endl;
	// cerr << "* Begin RDFVM initialization! *" << endl;
	// cerr << "*******************************" << endl;

	sim = _simulator;
	potts = _simulator->getPotts();
	automaton = potts->getAutomaton();
	xmlData = _xmlData;

	pUtils = sim->getParallelUtils();
	lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);

	cellInventory = & potts->getCellInventory();

	// Get useful plugins
	CC3d_log("Getting helpful plugins...");
	//cerr << "Getting helpful plugins..." << endl;

	bool pluginAlreadyRegisteredFlag;

	//		Get boundary pixel tracker plugin

	CC3d_log( "  Boundary pixel tracker plugin...");
	//cerr << "   Boundary pixel tracker plugin..." << endl;

	boundaryTrackerPlugin = (BoundaryPixelTrackerPlugin*)Simulator::pluginManager.get("BoundaryPixelTracker", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		CC3DXMLElement *BoundaryPixelTrackerXML = sim->getCC3DModuleData("Plugin", "BoundaryPixelTracker");
		boundaryTrackerPlugin->init(sim, BoundaryPixelTrackerXML);
	}

	//		Get cell type plugin
	CC3d_log("   Cell type plugin...");
	//cerr << "   Cell type plugin..." << endl;

	cellTypePlugin = (CellTypePlugin*)Simulator::pluginManager.get("CellType", &pluginAlreadyRegisteredFlag);
	ASSERT_OR_THROW("Cell type plugin must be registered for RDFVM, and in general.", pluginAlreadyRegisteredFlag);

	//		Get pixel tracker plugin
	
	pixelTrackerPlugin = (PixelTrackerPlugin*)Simulator::pluginManager.get("PixelTracker", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		CC3DXMLElement *pixelTrackerXML = sim->getCC3DModuleData("Plugin", "PixelTracker");
		pixelTrackerPlugin->init(sim, pixelTrackerXML);
	}
	
	fieldDim = potts->getCellFieldG()->getDim();

	boundaryStrategy = BoundaryStrategy::getInstance();

	maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

	// Get static inputs
	
	CC3d_log("Getting static RDFVM Solver inputs...");
	//cerr << "Getting static RDFVM Solver inputs..." << endl;

	//		Cell types

	CC3d_log("Getting cell types...");
	//cerr << "Getting cell types..." << endl;

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

	CC3d_log("Getting solver inputs...");
	//cerr << "Getting solver inputs..." << endl;

	CC3DXMLElement *el;

	//		Time discretization
	if (xmlData->findElement("DeltaT")) {
		el = xmlData->getFirstElement("DeltaT");
		incTime = (float)(el->getDouble());
		ASSERT_OR_THROW("FVM time increment must be greater than zero.", incTime > 0.0);
		if (el->findAttribute("unit")) { setUnitsTime(el->getAttribute("unit")); }

		CC3d_log("   Got time discretization: " << incTime << " " << getUnitsTime() << "/step");
		//cerr << "   Got time discretization: " << incTime << " " << getUnitsTime() << "/step" << endl;

	}

	//		Spatial discretization
	//		If 2D and z-length not specified, use unit length for z
	float DeltaX = lengthX;
	float DeltaY = lengthX;
	float DeltaZ = 1.0;
	if (xmlData->findElement("DeltaX")) {
		DeltaX = (float)(xmlData->getFirstElement("DeltaX")->getDouble());

		CC3d_log("   Got x-dimension discretization: " << DeltaX << " m");
		//cerr << "   Got x-dimension discretization: " << DeltaX << " m" << endl;

		ASSERT_OR_THROW("FVM spatial discretization must be greater than zero.", DeltaX > 0.0);
		if (xmlData->findElement("DeltaY")) {
			DeltaY = (float)(xmlData->getFirstElement("DeltaY")->getDouble());
			ASSERT_OR_THROW("FVM spatial discretization must be greater than zero.", DeltaY > 0.0);
		}
		else { DeltaY = DeltaX; }

		CC3d_log("   Got y-dimension discretization: " << DeltaY << " m");
		//cerr << "   Got y-dimension discretization: " << DeltaY << " m" << endl;

		if (xmlData->findElement("DeltaZ")) {
			DeltaZ = (float)(xmlData->getFirstElement("DeltaZ")->getDouble());
			ASSERT_OR_THROW("FVM spatial discretization must be greater than zero.", DeltaZ > 0.0);
		}
		else if (maxNeighborIndex > 3) { DeltaZ = DeltaX; }

		CC3d_log("   Got z-dimension discretization: " << DeltaZ << " m");
		//cerr << "   Got z-dimension discretization: " << DeltaZ << " m" << endl;

	}

	//		Diffusion fields

	CC3d_log("Getting diffusion fields...")
	//cerr << "Getting diffusion fields..." << endl;

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

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		el = fieldXMLVec[fieldIndex];
		ASSERT_OR_THROW("Each diffusion field must be given a name with the DiffusionField attribute Name", el->findAttribute("Name"));
		std::string fieldName = el->getAttribute("Name");

		CC3d_log("   Got field name: " << fieldName);
		//cerr << "   Got field name: " << fieldName << endl;

		// Check duplicates
		std::vector<std::string>::iterator fieldNameVec_itr = find(concentrationFieldNameVector.begin(), concentrationFieldNameVector.end(), fieldName);
		ASSERT_OR_THROW("Each FVM diffusion field must have a unique name", fieldNameVec_itr == concentrationFieldNameVector.end());

		CC3d_log("   Generating field wrap...");
		//cerr << "   Generating field wrap..." << endl;
		
		fieldNameToIndexMap.insert(make_pair(fieldName, fieldIndex));
		concentrationFieldNameVector[fieldIndex] = fieldName;
		concentrationFieldVector[fieldIndex] = new RDFVMField3DWrap<float>(this, fieldName);

		CC3d_log("   Registering field with Simulator...");
		//cerr << "   Registering field with Simulator..." << endl;

		sim->registerConcentrationField(fieldName, concentrationFieldVector[fieldIndex]);

		CC3DXMLElement *dData;
		CC3DXMLElement *dDataEl;
		
		// Diffusion data

		CC3d_log("   Getting diffusion data...");
		//cerr << "   Getting diffusion data..." << endl;

		ASSERT_OR_THROW("A DiffusionData element must be defined per FVM diffusion field", el->findElement("DiffusionData"));
		bool diffusionDefined = false;
		dData = el->getFirstElement("DiffusionData");

		//		Load medium diffusivity and mode if present
		dDataEl = dData->getFirstElement("DiffusionConstant");
		if (dDataEl) {
			diffusionDefined = true;
			constantDiffusionCoefficientsVec[fieldIndex] = dDataEl->getDouble();

			CC3d_log("   Got diffusion constant: " << constantDiffusionCoefficientsVec[fieldIndex] << " m2/s");
			//cerr << "   Got diffusion constant: " << constantDiffusionCoefficientsVec[fieldIndex] << " m2/s" << endl;

			useConstantDiffusivityBool[fieldIndex] = true;
		}
		//		Load diffusivity by type
		if (dData->findElement("DiffusivityByType")) {
			useConstantDiffusivityBool[fieldIndex] = false;
			useConstantDiffusivityByTypeBool[fieldIndex] = true;
			CC3d_log("   Got diffusivity by type.");
			//cerr << "   Got diffusivity by type." << endl;
		}
		//		Load diffusivity field in medium if present
		if (dData->findElement("DiffusivityFieldInMedium")) {
			if (diffusionDefined) { 
				CC3d_log("Warning: duplicate diffusion mode. Overwriting with DiffusivityFieldInMedium" );
				//cerr << "Warning: duplicate diffusion mode. Overwriting with DiffusivityFieldInMedium" << endl; 
				}
			else { 
				CC3d_log("   Got diffusivity field in medium. Diffusivity field is named: " + fieldName + diffusivityFieldSuffixStd);
				//cerr << "   Got diffusivity field in medium. Diffusivity field is named: " + fieldName + diffusivityFieldSuffixStd << endl;
				 }
			diffusionDefined = true;
			useConstantDiffusivityBool[fieldIndex] = false;
			useConstantDiffusivityByTypeBool[fieldIndex] = false;
			useFieldDiffusivityInMediumBool[fieldIndex] = true;
		}
		//		Load diffusivity field everywhere if present
		if (dData->findElement("DiffusivityFieldEverywhere")) {
			if (diffusionDefined) { 
				CC3d_log("Warning: duplicate diffusion mode. Overwriting with DiffusivityFieldEverywhere");
				//cerr << "Warning: duplicate diffusion mode. Overwriting with DiffusivityFieldEverywhere" << endl;
				 }
			else { 
				CC3d_log("   Got diffusivity field everywhere. Diffusivity field is named: " + fieldName + diffusivityFieldSuffixStd);
				//cerr << "   Got diffusivity field everywhere. Diffusivity field is named: " + fieldName + diffusivityFieldSuffixStd << endl; 
				}
			diffusionDefined = true;
			useConstantDiffusivityBool[fieldIndex] = false;
			useConstantDiffusivityByTypeBool[fieldIndex] = false;
			useFieldDiffusivityInMediumBool[fieldIndex] = false;
			useFieldDiffusivityEverywhereBool[fieldIndex] = true;
		}
		if (dData->findElement("InitialConcentrationExpression")) {
			initialExpressionStrings[fieldIndex] = dData->getFirstElement("InitialConcentrationExpression")->getText();
			CC3d_log("   Got initial concentration expression: " + initialExpressionStrings[fieldIndex]);
			//cerr << "   Got initial concentration expression: " + initialExpressionStrings[fieldIndex] << endl;
		}
		ASSERT_OR_THROW("A diffusion mode must be defined in DiffusionData.", diffusionDefined);

		//		Initialize cell type diffusivity coefficients as the same as for the field before loading type specifications
		constantDiffusionCoefficientsVecCellType[fieldIndex] = std::vector<double>(numCellTypes, constantDiffusionCoefficientsVec[fieldIndex]);

		//		Load all present cell type diffusion data, for future ref.
		//for each (CC3DXMLElement *typeData in dData->getElements("DiffusionCoefficient")) {
		for (CC3DXMLElement *typeData : dData->getElements("DiffusionCoefficient")) {

			std::string cellTypeName = typeData->getAttribute("CellType");
			std::map<std::string, unsigned int>::iterator cellTypeNameToIndexMap_itr = cellTypeNameToIndexMap.find(cellTypeName);
			if (cellTypeNameToIndexMap_itr != cellTypeNameToIndexMap.end()) {
				double typeDiffC = typeData->getDouble();
				constantDiffusionCoefficientsVecCellType[fieldIndex][cellTypeNameToIndexMap_itr->second] = typeDiffC;
				CC3d_log("   Got cell type (" << cellTypeName << ") diffusivity: " << typeDiffC << " m2/s");
				//cerr << "   Got cell type (" << cellTypeName << ") diffusivity: " << typeDiffC << " m2/s" << endl;
			}
		}

		//		Load all present cell type permeability data, for future ref.
		//			Interface permeation coefficients
		//for each (CC3DXMLElement *typeData in dData->getElements("PermIntCoefficient")) {
		for (CC3DXMLElement *typeData : dData->getElements("PermIntCoefficient")) {

			std::string cellType1Name = typeData->getAttribute("Type1");
			std::string cellType2Name = typeData->getAttribute("Type2");
			std::map<std::string, unsigned int>::iterator cellType1NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType1Name);
			std::map<std::string, unsigned int>::iterator cellType2NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType2Name);
			if (cellType1NameToIndexMap_itr != cellTypeNameToIndexMap.end() && cellType2NameToIndexMap_itr != cellTypeNameToIndexMap.end()) {
				double permC = typeData->getDouble();
				constantPermeationCoefficientsVecCellType[fieldIndex][cellType1NameToIndexMap_itr->second][cellType2NameToIndexMap_itr->second] = permC;
				constantPermeationCoefficientsVecCellType[fieldIndex][cellType2NameToIndexMap_itr->second][cellType1NameToIndexMap_itr->second] = permC;
				CC3d_log("   Got cell type (" << cellType1Name << ", " << cellType2Name << ") interface permeation coefficient: " << permC << " m/s");
				//cerr << "   Got cell type (" << cellType1Name << ", " << cellType2Name << ") interface permeation coefficient: " << permC << " m/s" << endl;
			}
		}
		//			Interface bias coefficients
		//for each (CC3DXMLElement *typeData in dData->getElements("PermIntBias")) {
		for (CC3DXMLElement *typeData : dData->getElements("PermIntBias")) {

			std::string cellType1Name = typeData->getAttribute("Type1");
			std::string cellType2Name = typeData->getAttribute("Type2");
			std::map<std::string, unsigned int>::iterator cellType1NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType1Name);
			std::map<std::string, unsigned int>::iterator cellType2NameToIndexMap_itr = cellTypeNameToIndexMap.find(cellType2Name);
			if (cellType1NameToIndexMap_itr != cellTypeNameToIndexMap.end() && cellType2NameToIndexMap_itr != cellTypeNameToIndexMap.end()) {
				double biasC = typeData->getDouble();
				constPermBiasCoeffsVecCellType[fieldIndex][cellType1NameToIndexMap_itr->second][cellType2NameToIndexMap_itr->second] = biasC;
				CC3d_log( "   Got cell type (" << cellType1Name << ", " << cellType2Name << ") interface bias coefficient: " << biasC);
				//cerr << "   Got cell type (" << cellType1Name << ", " << cellType2Name << ") interface bias coefficient: " << biasC << endl;
			}
		}

		//		Load simple permeable membranes if present
		usingSimplePermeableInterfaces[fieldIndex] = dData->findElement("SimplePermInt");
		if (usingSimplePermeableInterfaces[fieldIndex]) { fluxConditionInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::usePermeableSurfaces; }

		//		Load initial field expression if present
		useFieldInitialExprBool[fieldIndex] = dData->findElement("InitialConcentrationExpression");
		if (useFieldInitialExprBool[fieldIndex]) { fieldInitialExpr[fieldIndex] = dData->getFirstElement("InitialConcentrationExpression")->getData(); }

		// Reaction data
		CC3d_log("   Getting reaction data...");
		//cerr << "   Getting reaction data..." << endl;

		fieldExpressionStringsDiag[fieldIndex].clear();
		fieldExpressionStringsOffDiag[fieldIndex].clear();

		CC3DXMLElement *rData = el->getFirstElement("ReactionData");
		CC3DXMLElement *rDataEl;
		if (rData) {
			rDataEl = rData->getFirstElement("ExpressionSymbol");
			if (rDataEl) {
				fieldSymbolsVec[fieldIndex] = rDataEl->getText();
				CC3d_log( "   Got reaction expression symbol: " << fieldSymbolsVec[fieldIndex]);
				//cerr << "   Got reaction expression symbol: " << fieldSymbolsVec[fieldIndex] << endl;
			}

			if (rData->findElement("ExpressionMult")){
				//for each(CC3DXMLElement *expData in rData->getElements("ExpressionMult")) {
				for (CC3DXMLElement *expData : rData->getElements("ExpressionMult")) {

					std::string expStr = expData->getData();
					fieldExpressionStringsDiag[fieldIndex].push_back(expStr);
					CC3d_log("   Got multiplier reaction expression: " << expStr);
					//cerr << "   Got multiplier reaction expression: " << expStr << endl;
				}
			}

			if (rData->findElement("ExpressionIndep")) {
				//for each(CC3DXMLElement *expData in rData->getElements("ExpressionIndep")) {
				for (CC3DXMLElement *expData : rData->getElements("ExpressionIndep")) {

					std::string expStr = expData->getData();
					fieldExpressionStringsOffDiag[fieldIndex].push_back(expStr);
					CC3d_log("   Got independent reaction expression: " << expStr );
					//cerr << "   Got independent reaction expression: " << expStr << endl;
				}
			}
		}

		// Collect boundary conditions
		CC3d_log("   Collecting boundary conditions...");
		//cerr << "   Collecting boundary conditions..." << endl;

		CC3DXMLElement *bcData = el->getFirstElement("BoundaryConditions");
		if (bcData) { bcElementCollector.insert(make_pair(fieldName, bcData)); }
		
	}

	//		Assign reaction expression symbols for any field not already defined, for future ref
	for (unsigned int fieldIndex = 0; fieldIndex < fieldSymbolsVec.size(); ++fieldIndex) {
		if (fieldSymbolsVec[fieldIndex].size() == 0) {
			std::string fieldName = concentrationFieldNameVector[fieldIndex];
			fieldSymbolsVec[fieldIndex] = fieldName + expressionSuffixStd;
			CC3d_log( "   Assigning reaction expression symbol for " << fieldName << ": " << fieldSymbolsVec[fieldIndex]);
			//cerr << "   Assigning reaction expression symbol for " << fieldName << ": " << fieldSymbolsVec[fieldIndex] << endl;
		}
	}

	// Load diffusion initializers
	CC3d_log( "Loading diffusion initializers..." );
	//cerr << "Loading diffusion initializers..." << endl;

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		CC3d_log("   " << concentrationFieldNameVector[fieldIndex] << ": ");
		//cerr << "   " << concentrationFieldNameVector[fieldIndex] << ": ";

		if (useConstantDiffusivityBool[fieldIndex]) {
			CC3d_log( "constant diffusivity.");
			//cerr << "constant diffusivity." << endl;
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useConstantDiffusivity;
		}
		else if (useConstantDiffusivityByTypeBool[fieldIndex]) {
			CC3d_log("constant diffusivity by type.");
			//cerr << "constant diffusivity by type." << endl;
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useConstantDiffusivityByType;
		}
		else if (useFieldDiffusivityInMediumBool[fieldIndex]) {
			CC3d_log("diffusivity field in medium.");
			//cerr << "diffusivity field in medium." << endl;
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useFieldDiffusivityInMedium;
		}
		else if (useFieldDiffusivityEverywhereBool[fieldIndex]) {
			CC3d_log("diffusivity field everywhere.");
			//cerr << "diffusivity field everywhere." << endl;
			diffusivityModeInitializerPtrs[fieldIndex] = &ReactionDiffusionSolverFVM::useFieldDiffusivityEverywhere;
		}
	}
	
	// Build surface mappings
	// Note: will need updated for hex lattices
	CC3d_log("Building surface mappings..." );
	//cerr << "Building surface mappings..." << endl;

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
		CC3d_log( "   Processing surface for neighbor relative offset (" << offset.x << ", " << offset.y << ", " << offset.z << ") -> ");
		//cerr << "   Processing surface for neighbor relative offset (" << offset.x << ", " << offset.y << ", " << offset.z << ") -> ";
		if (offset.x > 0) {
			CC3d_log( "+x: " << nIdx );
			//cerr << "+x: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = 0;
			surfaceNormSign[nIdx] = 1;
			surfaceMapNameToIndex.insert(make_pair("MaxX", nIdx));
		}
		else if (offset.x < 0) {
			CC3d_log("-x: " << nIdx);
			//cerr << "-x: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = 0;
			surfaceNormSign[nIdx] = -1;
			surfaceMapNameToIndex.insert(make_pair("MinX", nIdx));
		}
		else if (offset.y > 0) {
			CC3d_log("+y: " << nIdx);
			//cerr << "+y: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = 1;
			surfaceNormSign[nIdx] = 1;
			surfaceMapNameToIndex.insert(make_pair("MaxY", nIdx));
		}
		else if (offset.y < 0) {
			CC3d_log("-y: " << nIdx);
			//cerr << "-y: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = 1;
			surfaceNormSign[nIdx] = -1;
			surfaceMapNameToIndex.insert(make_pair("MinY", nIdx));
		}
		else if (offset.z > 0) {
			CC3d_log("+z: " << nIdx);
			//cerr << "+z: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = 2;
			surfaceNormSign[nIdx] = 1;
			surfaceMapNameToIndex.insert(make_pair("MaxZ", nIdx));
		}
		else if (offset.z < 0) {
			CC3d_log( "-z: " << nIdx);
			//cerr << "-z: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = 2;
			surfaceNormSign[nIdx] = -1;
			surfaceMapNameToIndex.insert(make_pair("MinZ", nIdx));
		}
		else { // Assume an order
			CC3d_log("Warning: assuming a neighbor surface map: " << nIdx);
			//cerr << "Warning: assuming a neighbor surface map: " << nIdx << endl;

			indexMapSurfToCoord[nIdx] = indexMapSurfToCoordStd[nIdx];
			surfaceNormSign[nIdx] = surfaceNormSignStd[nIdx];
			surfaceMapNameToIndex.insert(make_pair(surfNameStd[nIdx], nIdx));
		}
	}
	setLengths(DeltaX, DeltaY, DeltaZ);

	// Load boundary conditions
	CC3d_log( "Loading boundary conditions...");
	//cerr << "Loading boundary conditions..." << endl;

	periodicBoundaryCheckVector = std::vector<bool>(3, false);
	std::vector<tuple<std::string, float> > basicBCDataFieldTemplate = std::vector<tuple<std::string, float> >(6, tuple<std::string, float>("ConstantDerivative", 0.0));
	basicBCData = std::vector<std::vector<tuple<std::string, float> > >(numFields, basicBCDataFieldTemplate);
	std::string boundaryName;
	boundaryName = potts->getBoundaryXName();
	if (boundaryName == "periodic") {
		CC3d_log("   Periodic x from Potts.");
		//cerr << "   Periodic x from Potts." << endl;

		periodicBoundaryCheckVector[0] = true;
		basicBCDataFieldTemplate[getSurfaceIndexByName("MaxX")] = tuple<std::string, float>("Periodic", 0.0);
		basicBCDataFieldTemplate[getSurfaceIndexByName("MinX")] = tuple<std::string, float>("Periodic", 0.0);
	}
	boundaryName = potts->getBoundaryYName();
	if (boundaryName == "periodic") {
		CC3d_log("   Periodic y from Potts.");
		//cerr << "   Periodic y from Potts." << endl;

		periodicBoundaryCheckVector[1] = true;
		basicBCDataFieldTemplate[getSurfaceIndexByName("MaxY")] = tuple<std::string, float>("Periodic", 0.0);
		basicBCDataFieldTemplate[getSurfaceIndexByName("MinY")] = tuple<std::string, float>("Periodic", 0.0);
	}
	if (maxNeighborIndex > 3) {
		boundaryName = potts->getBoundaryZName();
		if (boundaryName == "periodic") {
			CC3d_log( "   Periodic z from Potts.");
			//cerr << "   Periodic z from Potts." << endl;

			periodicBoundaryCheckVector[2] = true;
			basicBCDataFieldTemplate[getSurfaceIndexByName("MaxZ")] = tuple<std::string, float>("Periodic", 0.0);
			basicBCDataFieldTemplate[getSurfaceIndexByName("MinZ")] = tuple<std::string, float>("Periodic", 0.0);
		}
	}

	std::map<std::string, CC3DXMLElement *>::iterator bc_itr;
	for (unsigned int fieldIndex = 0; fieldIndex < concentrationFieldNameVector.size(); ++fieldIndex) {
		std::string fieldName = concentrationFieldNameVector[fieldIndex];
		std::vector<tuple<std::string, float> > basicBCDataField = basicBCDataFieldTemplate;
		CC3d_log("Loading boundary conditions for field " + fieldName);
		//cerr << "Loading boundary conditions for field " + fieldName << endl;

		bc_itr = bcElementCollector.find(fieldName);
		if (bc_itr != bcElementCollector.end()) {
			CC3DXMLElementList bcPlaneElementList = bc_itr->second->getElements("Plane");
			//for each (CC3DXMLElement *bcEl in bcPlaneElementList) {
			for (CC3DXMLElement *bcEl : bcPlaneElementList) {

				std::string axisString = bcEl->getAttribute("Axis");
				//for each (CC3DXMLElement *bcElSpec in bcEl->children) {
				for (CC3DXMLElement *bcElSpec : bcEl->children) {

					std::string posString = bcElSpec->getAttribute("PlanePosition");
					std::string surfaceName = posString + axisString;
					unsigned int surfaceIndex = getSurfaceIndexByName(surfaceName);
					unsigned int dimIndex = getIndexSurfToCoord(surfaceIndex);
					ASSERT_OR_THROW("Cannot specify a boundary condition for a periodic boundary.", !periodicBoundaryCheckVector[dimIndex]);

					std::string bcTypeName = bcElSpec->getName();
					ASSERT_OR_THROW(std::string("Unknown boundary condition type: " + bcTypeName + ". Valid inputs are ConstantValue and ConstantDerivative"), 
						bcTypeName == "ConstantValue" || bcTypeName == "ConstantDerivative");
					float bcVal = (float)(bcElSpec->getAttributeAsDouble("PlanePosition"));
					CC3d_log("   Got boundary condition " << bcTypeName << " for " << surfaceName << " with value " << bcVal);
					CC3d_log("      Loading to surface index " << surfaceIndex);
					CC3d_log( " for dimension index " << dimIndex);
					// cerr << "   Got boundary condition " << bcTypeName << " for " << surfaceName << " with value " << bcVal << endl;
					// cerr << "      Loading to surface index " << surfaceIndex;
					// cerr << " for dimension index " << dimIndex << endl;
					
					basicBCDataField[surfaceIndex] = tuple<std::string, float>(bcTypeName, bcVal);
				}
			}
		}
		basicBCData[fieldIndex] = basicBCDataField;
	}

	// Initialize reaction expressions
	CC3d_log("Initializing reaction expressions...");
	//cerr << "Initializing reaction expressions..." << endl;

	fieldExpressionStringsMergedDiag = std::vector<std::string>(numFields, "");
	fieldExpressionStringsMergedOffDiag = std::vector<std::string>(numFields, "");
	
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
		CC3d_log("Constructing reaction expressions for " + concentrationFieldNameVector[fieldIndex]);
		//cerr << "Constructing reaction expressions for " + concentrationFieldNameVector[fieldIndex] << endl;

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
			CC3d_log("   Multiplier function: " + expDiag);
			//cerr << "   Multiplier function: " + expDiag << endl;
		}

		if (fieldExpressionStringsOffDiag[fieldIndex].size() == 0) { expOffDiag = "0.0"; }
		else {
			expOffDiag = fieldExpressionStringsOffDiag[fieldIndex][0];
			if (fieldExpressionStringsOffDiag[fieldIndex].size() > 1) {
				for (unsigned int expIndex = 1; expIndex < fieldExpressionStringsOffDiag[fieldIndex].size(); ++expIndex) {
					expOffDiag += "+" + fieldExpressionStringsOffDiag[fieldIndex][expIndex];
				}
			}
			CC3d_log("   Independent function: " + expOffDiag);
			//cerr << "   Independent function: " + expOffDiag << endl;
		}

		if (expDiag.size() == 0) { expDiag = "0.0"; }
		if (expOffDiag.size() == 0) { expOffDiag = "0.0"; }
		fieldExpressionStringsMergedDiag[fieldIndex] = expDiag;
		fieldExpressionStringsMergedOffDiag[fieldIndex] = expOffDiag;
	}

	// Build lattice
	
	CC3d_log("Building lattice...");
	//cerr << "Building lattice..." << endl;

	initializeFVs(fieldDim);

	// Initialize concentrations

	CC3d_log("Initializing concentrations...");
	//cerr << "Initializing concentrations..." << endl;

	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
		if (useFieldInitialExprBool[fieldIndex]) { initializeFieldUsingEquation(fieldIndex, fieldInitialExpr[fieldIndex]); }

	//		Auto time stepping
	autoTimeSubStep = _xmlData->findElement("AutoTimeSubStep");
	if (autoTimeSubStep) {
		CC3d_log("RDVFM got automatic time sub-stepping.");
		 //cerr << "RDVFM got automatic time sub-stepping." << endl; 
		 }
	// replace with vector
	fvMaxStableTimeSteps = new std::vector<double>;
	//fvMaxStableTimeSteps = new concurrency::concurrent_vector<double>;

	//		Simple mass conservation option
	simpleMassConservation = _xmlData->findElement("SimpleMassConservation");
	if (simpleMassConservation) { 
		CC3d_log("RDVFM got simple mass conservation.");
		//cerr << "RDVFM got simple mass conservation." << endl;

		pixelTrackerPlugin->enableMediumTracker(true);
		pixelTrackerPlugin->enableFullInitAtStart(true);
	}

	concentrationVecCopiesMedium = std::vector<double>(numFields, 0.0);
	massConsCorrectionFactorsMedium = std::vector<double>(numFields, 1.0);

	CC3d_log("Registering RDFVM Solver...");
	//cerr << "Registering RDFVM Solver..." << endl;

	potts->getCellFactoryGroupPtr()->registerClass(&ReactionDiffusionSolverFVMCellDataAccessor);
	potts->registerCellGChangeWatcher(this);
	sim->registerSteerableObject(this);

	CC3d_log("*****************************");
	CC3d_log("* End RDFVM initialization! *");
	CC3d_log("*****************************");
	// cerr << "*****************************" << endl;
	// cerr << "* End RDFVM initialization! *" << endl;
	// cerr << "*****************************" << endl;

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

		if (simpleMassConservation) {
			CC3d_log("RDFVM Initializing pixel tracker data...");
			//cerr << "RDFVM Initializing pixel tracker data..." << endl;

			pixelTrackerPlugin->enableMediumTracker(true);
			pixelTrackerPlugin->enableFullInitAtStart(true);
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

	CC3d_log("RDFVM Step begin...");
	//cerr << "RDFVM Step begin..." << endl;
	
	auto &_fieldFVs = fieldFVs;

	if (simpleMassConservation) {

		CC3d_log( "   Simple mass conservation: updating fields...");
		//cerr << "   Simple mass conservation: updating fields..." << endl;

			
		#pragma omp parallel for shared (fieldFVs)
		for (int i=0;i<fieldFVs.size();i++){
			CellG *cell = this->FVtoCellMap(fieldFVs[i]);
			std::vector<double> correctionFactors;
			if (cell) { correctionFactors = this->getReactionDiffusionSolverFVMCellDataAccessorPtr()->get(cell->extraAttribPtr)->massConsCorrectionFactors; }
			else { correctionFactors = this->massConsCorrectionFactorsMedium; }
			// fv->setConcentrationVec(vecMult<double>()(correctionFactors, fv->getConcentrationVec()));
			for (unsigned int fieldIndex = 0; fieldIndex < correctionFactors.size(); ++fieldIndex)
				fieldFVs[i]->setConcentration(fieldIndex, fieldFVs[i]->getConcentration(fieldIndex) * correctionFactors[fieldIndex]);
		}
		

	}

	CC3d_log("   Explicit RD integration...");
	//cerr << "   Explicit RD integration..." << endl;

	double intTime = 0.0;

	fieldDim = potts->getCellFieldG()->getDim();
	auto _fieldDim = fieldDim;

	///TODO: Ask Dr. Sego about why grow_to_at_least was removed? Or maybe it was moved to another module?
	// if (autoTimeSubStep) { fvMaxStableTimeSteps->grow_to_at_least(fieldDim.x*fieldDim.y*fieldDim.z); }
	// else { fvMaxStableTimeSteps = 0; }
	fvMaxStableTimeSteps = 0;
	while (intTime < incTime) {

		if (autoTimeSubStep) {

			CC3d_log( "      Integrating with maximum stable time step... ");
			//cerr << "      Integrating with maximum stable time step... ";
			
			
			#pragma omp parallel for shared (fieldDim)
			for (int fieldIndex=0;fieldIndex<fieldDim.x*fieldDim.y*fieldDim.z;fieldIndex++){
				fvMaxStableTimeSteps->at(fieldIndex) = this->getFieldFV(fieldIndex)->solveStable();
			}
		
			CC3d_log("calculating maximum stable time step... ");
			//cerr << "calculating maximum stable time step... ";

			// Might be more efficient using a combinable
			integrationTimeStep = min(*min_element(fvMaxStableTimeSteps->begin(), fvMaxStableTimeSteps->end()), incTime - intTime);
		}
		else { 

			CC3d_log("      Integrating with fixed time step... ");
			//cerr << "      Integrating with fixed time step... ";

			integrationTimeStep = incTime - intTime;

			#pragma omp parallel for shared (fieldFVs)
			for (int i=0;i< fieldFVs.size();i++){
				fieldFVs[i]->solve();
			}

		}

		CC3d_log(integrationTimeStep << " s." );
		//cerr << integrationTimeStep << " s." << endl;

		CC3d_log("      Updating... ");
		//cerr << "      Updating... ";
		
		#pragma omp parallel for shared (fieldFVs)
		for (int i=0;i<fieldFVs.size();i++){
			fieldFVs[i]->update(this->getIntegrationTimeStep());
		}

		intTime += integrationTimeStep;
		physTime += integrationTimeStep;

		CC3d_log("done: " << physTime / unitTimeConv << " " << getUnitsTime());
		//cerr << "done: " << physTime / unitTimeConv << " " << getUnitsTime() << endl;

	}

	if (simpleMassConservation) {

		CC3d_log("   Simple mass conservation: updating properties..." );
		//cerr << "   Simple mass conservation: updating properties..." << endl;

		updateTotalConcentrations();

	}

	CC3d_log("RDFVM Step complete.");
	//cerr << "RDFVM Step complete." << endl;

	pUtils->unsetLock(lockPtr);

}

////////////////////////////////////////////////////////// Steerable interface ///////////////////////////////////////////////////////////
void ReactionDiffusionSolverFVM::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

	// Get ECMaterials plugin
	//		Requires cell type diffusion specification for each field using material-dependent diffusion
	ASSERT_OR_THROW("ECMaterials plugin not currently supported in FVM solver.", !usingECMaterials);
	bool pluginAlreadyRegisteredFlag = false;
	if (usingECMaterials) {
		// ecMaterialsPlugin = (ECMaterialsPlugin*)Simulator::pluginManager.get("ECMaterials", &pluginAlreadyRegisteredFlag);
		ASSERT_OR_THROW("ECMaterials plugin must be registered before usage by FVM solver.", pluginAlreadyRegisteredFlag);
	}

	// Get steerable inputs
	//		Auto time stepping
	autoTimeSubStep = _xmlData->findElement("AutoTimeSubStep");
	if (autoTimeSubStep) {
		CC3d_log("RDVFM got automatic time sub-stepping.");
		 //cerr << "RDVFM got automatic time sub-stepping." << endl; 
		 }

	//		Simple mass conservation option
	bool simpleMassConservation_old = simpleMassConservation;
	simpleMassConservation = _xmlData->findElement("SimpleMassConservation");
	if (!simpleMassConservation_old && simpleMassConservation) {

		CC3d_log("RDFVM enabling simple mass conservation...");
		//cerr << "RDFVM enabling simple mass conservation..." << endl;

		if (!(pixelTrackerPlugin->fullyInitialized() && pixelTrackerPlugin->trackingMedium())) {

			CC3d_log("RDFVM Initializing pixel tracker data...");
			//cerr << "RDFVM Initializing pixel tracker data..." << endl;

			pixelTrackerPlugin->enableMediumTracker(true);
			pixelTrackerPlugin->fullTrackerDataInit();

		}
	}
	else if (simpleMassConservation_old && !simpleMassConservation) {
		CC3d_log("RDFVM disabling simple mass conservation...");
		 //cerr << "RDFVM disabling simple mass conservation..." << endl; 
		 }

}

std::string ReactionDiffusionSolverFVM::steerableName() {
	return toString();
}

std::string ReactionDiffusionSolverFVM::toString() {
	return "ReactionDiffusionSolverFVM";
}

////////////////////////////////////////////////////// Cell field watcher interface //////////////////////////////////////////////////////
void ReactionDiffusionSolverFVM::field3DAdditionalPt(const Point3D &ptAdd) { (this->*field3DAdditionalPtFcnPtr)(ptAdd); }

// Prevents erroneous calls to CellGChangeWatcher during initialization routines
void ReactionDiffusionSolverFVM::field3DAdditionalPtPreStartup(const Point3D &ptAdd) {
	flipSourcePt = ptAdd;
	field3DAdditionalPtFcnPtr = &ReactionDiffusionSolverFVM::field3DAdditionalPtPostStartup;
	field3DChangeFcnPtr = &ReactionDiffusionSolverFVM::field3DChangePostStartup;
}

void ReactionDiffusionSolverFVM::field3DAdditionalPtPostStartup(const Point3D &ptAdd) { flipSourcePt = ptAdd; }

void ReactionDiffusionSolverFVM::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) { (this->*field3DChangeFcnPtr)(pt, newCell, oldCell); }

void ReactionDiffusionSolverFVM::field3DChangePostStartup(const Point3D &pt, CellG *newCell, CellG *oldCell) {

	if (simpleMassConservation && newCell != oldCell) {

		// Load cell data just in time if necessary
		if (!cellDataLoaded) { loadCellData(); }

		// Get concentrations at copy-associated sites
		std::vector<double> concentrationVecOld = getFieldFV(pt)->getConcentrationVec();
		std::vector<double> concentrationVecNew = getFieldFV(flipSourcePt)->getConcentrationVec();

		if (oldCell) {

			// Get cell data
			std::vector<double> &concentrationVecCopiesOld = ReactionDiffusionSolverFVMCellDataAccessor.get(oldCell->extraAttribPtr)->concentrationVecCopies;
			std::vector<double> &massConsCorrectionFactorsOld = ReactionDiffusionSolverFVMCellDataAccessor.get(oldCell->extraAttribPtr)->massConsCorrectionFactors;

			// On-the-fly initializations
			if (concentrationVecCopiesOld.size() < numFields) { concentrationVecCopiesOld = totalCellConcentration(oldCell); }
			if (massConsCorrectionFactorsOld.size() < numFields) { massConsCorrectionFactorsOld = std::vector<double>(numFields, 1.0); }

			// Update correction factors
			for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) {
				if (concentrationVecCopiesOld[fieldIndex] * massConsCorrectionFactorsOld[fieldIndex] > 0.0) {
					double den = 1 / massConsCorrectionFactorsOld[fieldIndex] - concentrationVecOld[fieldIndex] / concentrationVecCopiesOld[fieldIndex];
					if (den > 0.0) { massConsCorrectionFactorsOld[fieldIndex] = 1 / den; }
				}
			}

		}
		else {

			// Update correction factors
			for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex){
				if (concentrationVecCopiesMedium[fieldIndex] * massConsCorrectionFactorsMedium[fieldIndex] > 0.0) {
					double den = 1 / massConsCorrectionFactorsMedium[fieldIndex] - concentrationVecOld[fieldIndex] / concentrationVecCopiesMedium[fieldIndex];
					if (den > 0.0) { massConsCorrectionFactorsMedium[fieldIndex] = 1 / den; }
				}
			}

		}

		if (newCell) {

			// Get cell data
			std::vector<double> &concentrationVecCopiesNew = ReactionDiffusionSolverFVMCellDataAccessor.get(newCell->extraAttribPtr)->concentrationVecCopies;
			std::vector<double> &massConsCorrectionFactorsNew = ReactionDiffusionSolverFVMCellDataAccessor.get(newCell->extraAttribPtr)->massConsCorrectionFactors;

			// On-the-fly initializations
			if (concentrationVecCopiesNew.size() < numFields) { concentrationVecCopiesNew = totalCellConcentration(newCell); }
			if (massConsCorrectionFactorsNew.size() < numFields) { massConsCorrectionFactorsNew = std::vector<double>(numFields, 1.0); }

			// Update correction factors
			for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
				if (concentrationVecCopiesNew[fieldIndex] * massConsCorrectionFactorsNew[fieldIndex] > 0.0)
					massConsCorrectionFactorsNew[fieldIndex] = 1 / (1 / massConsCorrectionFactorsNew[fieldIndex] + concentrationVecNew[fieldIndex] / concentrationVecCopiesNew[fieldIndex]);

		}
		else {

			// Update correction factors
			for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex)
				if (concentrationVecCopiesMedium[fieldIndex] * massConsCorrectionFactorsMedium[fieldIndex] > 0.0)
					massConsCorrectionFactorsMedium[fieldIndex] = 1 / (1 / massConsCorrectionFactorsMedium[fieldIndex] + concentrationVecNew[fieldIndex] / concentrationVecCopiesMedium[fieldIndex]);

		}

		getFieldFV(pt)->setConcentrationVec(concentrationVecNew);

	}
}

///////////////////////////////////////////////////////////// Solver routines ////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFVM::loadCellData() {

	cellInventory = &potts->getCellInventory();
	CC3d_log("RDFVM Initializing cell data...");
	//cerr << "RDFVM Initializing cell data..." << endl;

	initializeCellData(numFields);
	CC3d_log("RDFVM Loading cell data...");
	//cerr << "RDFVM Loading cell data..." << endl;

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
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){
		loadFieldExpressionMultiplier(_fieldIndex, fieldFVs[i]);
	}
	//parallel_for_each(fieldFVs->begin(), fieldFVs->end(), [&](ReactionDiffusionSolverFV *fv) { loadFieldExpressionMultiplier(_fieldIndex, fv); });
}

void ReactionDiffusionSolverFVM::loadFieldExpressionMultiplier(std::string _fieldName, std::string _expr) {
	setFieldExpressionMultiplier(_fieldName, _expr);
	loadFieldExpressionMultiplier(getFieldIndexByName(_fieldName));
}

void ReactionDiffusionSolverFVM::loadFieldExpressionIndependent(unsigned int _fieldIndex) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){
				loadFieldExpressionIndependent(_fieldIndex, fieldFVs[i]); 
		}
	//parallel_for_each(fieldFVs->begin(), fieldFVs->end(), [&](ReactionDiffusionSolverFV *fv) { loadFieldExpressionIndependent(_fieldIndex, fv); });
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

	//fieldFVs = new concurrency::concurrent_vector<ReactionDiffusionSolverFV*>(_fieldDim.x*_fieldDim.y*_fieldDim.z);
    fieldFVs = std::vector<ReactionDiffusionSolverFV*>(_fieldDim.x*_fieldDim.y*_fieldDim.z);

	CC3d_log("Constructing lattice with " << _fieldDim.x*_fieldDim.y*_fieldDim.z << " sites...");
	//cerr << "Constructing lattice with " << _fieldDim.x*_fieldDim.y*_fieldDim.z << " sites..." << endl;


    // replaced parallel_for from parallels to openmp's parallel for implementation
    #pragma omp parallel for shared (_fieldDim)
	for (int ind=0;ind<_fieldDim.x*_fieldDim.y*_fieldDim.z;ind++){
		ReactionDiffusionSolverFV *fv = new ReactionDiffusionSolverFV(this, ind2pt(ind), (int)this->getConcentrationFieldNameVector().size());
		this->setFieldFV(ind, fv);
	}
    
	CC3d_log("Initializing FVs...");
	//cerr << "Initializing FVs..." << endl;


    #pragma omp parallel for shared (fieldFVs)
	for (int i=0;i< fieldFVs.size();i++){ 
		fieldFVs[i]->initialize();
	}
	CC3d_log("Setting field symbols...");
	//cerr << "Setting field symbols..." << endl;
	
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){
	for (int fieldIndex=0;fieldIndex<fieldFVs[i]->getConcentrationVec().size();++fieldIndex){ 
			fieldFVs[i]->registerFieldSymbol(fieldIndex, this->getFieldSymbol(fieldIndex));
		}
	}
	
	CC3d_log("Loading field expressions...");
	//cerr << "Loading field expressions..." << endl;

	loadFieldExpressions();

	CC3d_log("Setting initial FV diffusivity method...");
	//cerr << "Setting initial FV diffusivity method..." << endl;

	// Diffusion mode initializations
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { (this->*diffusivityModeInitializerPtrs[fieldIndex])(fieldIndex); }

	// Lattice-wide flux condition initializations
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { (this->*fluxConditionInitializerPtrs[fieldIndex])(fieldIndex); }

	// Apply basic boundary conditions

	CC3d_log("Applying basic boundary conditions...");
	//cerr << "Applying basic boundary conditions..." << endl;

	unsigned int surfaceIndex;
	ReactionDiffusionSolverFV *fv;
	unsigned int fieldIndex;
	std::vector<std::tuple<short, std::string> > bLocSpecs = std::vector<std::tuple<short, std::string> >(2, std::tuple<unsigned int, std::string>(0, ""));

	//		z boundaries

	if (maxNeighborIndex > 3) {

		CC3d_log("   along z-boundaries...");
		//cerr << "   along z-boundaries..." << endl;

		bLocSpecs[0] = tuple<short, std::string>(0, "MinZ");
		bLocSpecs[1] = tuple<short, std::string>(fieldDim.z - 1, "MaxZ");
		for (short x = 0; x < fieldDim.x; ++x)
			for (short y = 0; y < fieldDim.y; ++y)
				//for each (tuple<unsigned int, std::string> bLocSpec in bLocSpecs) {
				for (const auto& bLocSpec: bLocSpecs){
					short z = std::get<0>(bLocSpec);
					surfaceIndex = getSurfaceIndexByName(std::get<1>(bLocSpec));
					fv = getFieldFV(Point3D(x, y, z));
					//for each(std::string fieldName in concentrationFieldNameVector) {
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

	CC3d_log( "   along y-boundaries...");
	//cerr << "   along y-boundaries..." << endl;

	bLocSpecs[0] = tuple<short, std::string>(0, "MinY");
	bLocSpecs[1] = tuple<short, std::string>(fieldDim.y - 1, "MaxY");
	for (short x = 0; x < fieldDim.x; ++x)
		for (short z = 0; z < fieldDim.z; ++z)
			//for each (tuple<unsigned int, std::string> bLocSpec in bLocSpecs) {
			for(const auto& bLocSpec : bLocSpecs){
				short y = std::get<0>(bLocSpec);
				surfaceIndex = getSurfaceIndexByName(std::get<1>(bLocSpec));
				fv = getFieldFV(Point3D(x, y, z));
				//for each(std::string fieldName in concentrationFieldNameVector) {
				for(const auto& fieldName : concentrationFieldNameVector){
					fieldIndex = getFieldIndexByName(fieldName);
					std::string bcName = std::get<0>(basicBCData[fieldIndex][surfaceIndex]);
					float bcVal = std::get<1>(basicBCData[fieldIndex][surfaceIndex]);
					if (bcName == "ConstantValue") { fv->useFixedConcentration(fieldIndex, surfaceIndex, bcVal); }
					else if (bcName == "ConstantDerivative") { fv->useFixedFluxSurface(fieldIndex, surfaceIndex, bcVal); }
				}
			}

	//		x boundaries

	CC3d_log("   along x-boundaries...");
	//cerr << "   along x-boundaries..." << endl;

	bLocSpecs[0] = tuple<short, std::string>(0, "MinX");
	bLocSpecs[1] = tuple<short, std::string>(fieldDim.x - 1, "MaxX");
	for (short y = 0; y < fieldDim.y; ++y)
		for (short z = 0; z < fieldDim.z; ++z)
			//for each (tuple<unsigned int, std::string> bLocSpec in bLocSpecs) {
			for(const auto& bLocSpec : bLocSpecs){
				short x = std::get<0>(bLocSpec);
				surfaceIndex = getSurfaceIndexByName(std::get<1>(bLocSpec));
				fv = getFieldFV(Point3D(x, y, z));
				//for each(std::string fieldName in concentrationFieldNameVector) {
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
		CC3d_log(e.GetMsg());
		//cerr << e.GetMsg() << endl;
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}

std::vector<double> ReactionDiffusionSolverFVM::totalCellConcentration(const CellG *_cell) { return totalPixelSetConcentration(getCellPixelVec(_cell)); }

std::vector<Point3D> ReactionDiffusionSolverFVM::getCellPixelVec(const CellG *_cell) {
	std::set<PixelTrackerData> &pixelSet = pixelTrackerPlugin->getPixelTrackerAccessorPtr()->get(_cell->extraAttribPtr)->pixelSet;
	std::vector<Point3D> pixelVec = std::vector<Point3D>(pixelSet.size());
	unsigned int ptdIndex = 0;
	for (set<PixelTrackerData >::iterator sitr = pixelSet.begin(); sitr != pixelSet.end(); ++sitr) {
		pixelVec[ptdIndex] = sitr->pixel;
		++ptdIndex;
	}
	return pixelVec;
}

std::vector<double> ReactionDiffusionSolverFVM::totalMediumConcentration() {
	// Load pixels for parallel sweep of medium

	std::vector<Point3D> pixelVec = getMediumPixelVec();
	std::vector<Point3D> pixelVecPar = std::vector<Point3D>(pixelVec.size());
	for (unsigned int pixelIndex = 0; pixelIndex < pixelVec.size(); ++pixelIndex) { pixelVecPar[pixelIndex] = pixelVec[pixelIndex]; }

	//concurrency::concurrent_vector<Point3D> pixelVecPar = concurrency::concurrent_vector<Point3D>(pixelVec.size());

	// Calculate total concentrations on each thread
		
	std::vector<double> sumEl(numFields,0.0);
	int n_threads = omp_get_num_threads();
	std::vector<vector<double>> res (n_threads,std::vector<double>(numFields, 0.0));
	
	#pragma omp parallel for shared(pixelVecPar,res)
	for (int i=0;i<pixelVecPar.size();i++){ 
		int index = omp_get_thread_num();
		res[index] = vecPlus<double>()(res[index],std::vector<double>(this->getFieldFV(pixelVecPar[i])->getConcentrationVec()));
	}

	// //reduction
	std::vector<double> res_reduced = std::vector<double>(numFields, 0.0);
	#pragma omp critical
	for (int i=0;i<n_threads;i++){ 
		res_reduced = vecPlus<double>()(res[i],res_reduced);
	}


	// Reset mass conservation correction factors
	massConsCorrectionFactorsMedium = std::vector<double>(numFields, 1.0);

	return res_reduced;
}

void ReactionDiffusionSolverFVM::updateTotalCellConcentrations() {
	unsigned int numCells = (unsigned int)(cellInventory->getSize());
	if (numCells > 0) {
		// Load cell pointers for parallel update
		
		std::vector<CellG* > vecCells = std::vector<CellG* >(numCells);
		unsigned int cellIdx = 0;


		for (CellInventory::cellInventoryIterator cItr = cellInventory->cellInventoryBegin(); cItr != cellInventory->cellInventoryEnd(); ++cItr) {
			vecCells[cellIdx] = cItr->second;
			++cellIdx;
		}

		// Perform parallel update
		#pragma omp parallel for shared (vecCells)
		for (int i=0;i<vecCells.size();i++){ 
			BasicClassAccessor<ReactionDiffusionSolverFVMCellData> *cellData = this->getReactionDiffusionSolverFVMCellDataAccessorPtr();
			std::vector<double> &concentrationVecCopies = cellData->get(vecCells[i]->extraAttribPtr)->concentrationVecCopies;
			std::vector<double> &massConsCorrectionFactors = cellData->get(vecCells[i]->extraAttribPtr)->massConsCorrectionFactors;
			concentrationVecCopies = this->totalCellConcentration(vecCells[i]);
			massConsCorrectionFactors = std::vector<double>(concentrationVecCopies.size(), 1.0);
		}


	}
}

void ReactionDiffusionSolverFVM::updateTotalConcentrations() {
	if (!(pixelTrackerPlugin->fullyInitialized())) { pixelTrackerPlugin->fullTrackerDataInit(); }

	// Update total mass in medium
	concentrationVecCopiesMedium = totalMediumConcentration();

	// Update total mass in cells
	updateTotalCellConcentrations();
}

std::vector<Point3D> ReactionDiffusionSolverFVM::getMediumPixelVec() {
	std::set<PixelTrackerData> pixelSet = pixelTrackerPlugin->getMediumPixelSet();
	std::vector<Point3D> pixelVec = std::vector<Point3D>(pixelSet.size());
	unsigned int ptdIndex = 0;
	for (set<PixelTrackerData >::iterator sitr = pixelSet.begin(); sitr != pixelSet.end(); ++sitr) {
		pixelVec[ptdIndex] = sitr->pixel;
		++ptdIndex;
	}
	return pixelVec;
}

std::vector<double> ReactionDiffusionSolverFVM::totalPixelSetConcentration(std::vector<Point3D> _pixelVec) {
	std::vector<double> sumVec = std::vector<double>(numFields, 0.0);
	for (unsigned int pixelIndex = 0; pixelIndex < _pixelVec.size(); ++pixelIndex) {
		for (unsigned int vecIndex = 0; vecIndex < numFields; ++vecIndex) { sumVec[vecIndex] += getFieldFV(_pixelVec[pixelIndex])->getConcentrationOld(vecIndex); }
	}
	return sumVec;
}


////////////////////////////////////////////////////////////// FV interface //////////////////////////////////////////////////////////////

Point3D ReactionDiffusionSolverFVM::getCoordsOfFV(ReactionDiffusionSolverFV *_fv) { return _fv->getCoords(); }

CellG * ReactionDiffusionSolverFVM::FVtoCellMap(ReactionDiffusionSolverFV * _fv) { return potts->getCellFieldG()->get(_fv->getCoords()); }

void ReactionDiffusionSolverFVM::useConstantDiffusivity(unsigned int _fieldIndex, double _diffusivityCoefficient) {	
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){ 
		fieldFVs[i]->useConstantDiffusivity(_fieldIndex, _diffusivityCoefficient);
	}
}
void ReactionDiffusionSolverFVM::useConstantDiffusivityByType(unsigned int _fieldIndex, double _diffusivityCoefficient) {
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){ 
		fieldFVs[i]->useConstantDiffusivityById(_fieldIndex, _diffusivityCoefficient);
	}
}
void ReactionDiffusionSolverFVM::useFieldDiffusivityInMedium(unsigned int _fieldIndex) {
	initDiffusivityField(_fieldIndex);
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){ 
		fieldFVs[i]->useFieldDiffusivityInMedium(_fieldIndex);
	}
}
void ReactionDiffusionSolverFVM::useFieldDiffusivityEverywhere(unsigned int _fieldIndex) {
	initDiffusivityField(_fieldIndex);
	auto &_fieldFVs = fieldFVs;
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){ 
		fieldFVs[i]->useFieldDiffusivityEverywhere(_fieldIndex);
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

void ReactionDiffusionSolverFVM::useDiffusiveSurfaces(unsigned int _fieldIndex) {
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){ 
		fieldFVs[i]->useDiffusiveSurfaces(_fieldIndex);
	}
}

void ReactionDiffusionSolverFVM::usePermeableSurfaces(unsigned int _fieldIndex, bool _activate) {
	#pragma omp parallel for shared (fieldFVs)
	for (int i=0;i<fieldFVs.size();i++){ 
		fieldFVs[i]->usePermeableSurfaces(_fieldIndex, _activate);
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
	if (simpleMassConservation) { 
		cellData->concentrationVecCopies = std::vector<double>(totalCellConcentration(_cell));
		cellData->massConsCorrectionFactors = std::vector<double>(_numFields, 1.0);
	}
}

void ReactionDiffusionSolverFVM::initializeCellData(unsigned int _numFields) {
	if (!(pixelTrackerPlugin->fullyInitialized())) { pixelTrackerPlugin->fullTrackerDataInit(); }

	for (cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr)
		initializeCellData(cellInventory->getCell(cell_itr), _numFields);

	if (simpleMassConservation) { concentrationVecCopiesMedium = totalMediumConcentration(); }
}

double ReactionDiffusionSolverFVM::getCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex) {

	ASSERT_OR_THROW("Inappropriate call to medium coefficient.", _cell);

	ReactionDiffusionSolverFVMCellData *cellData = ReactionDiffusionSolverFVMCellDataAccessor.get(_cell->extraAttribPtr);
	std::vector<double> &diffusivityCoefficients = cellData->diffusivityCoefficients;
	if (diffusivityCoefficients.size() < _fieldIndex) { 

		CC3d_log("Initializing cell diffusivity coefficient on the fly...");
		//cerr << "Initializing cell diffusivity coefficient on the fly..." << endl;

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
	for (cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr) {
		setCellDiffusivityCoefficient(cellInventory->getCell(cell_itr), _fieldIndex);
	}
}

void ReactionDiffusionSolverFVM::setCellDiffusivityCoefficients() {
	for (unsigned int fieldIndex = 0; fieldIndex < numFields; ++fieldIndex) { setCellDiffusivityCoefficients(fieldIndex); }
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
	for (cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr)
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
	for (cell_itr = cellInventory->cellInventoryBegin(); cell_itr != cellInventory->cellInventoryEnd(); ++cell_itr)
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
}

//////////////////////////////////////////////////////////// Solver functions ////////////////////////////////////////////////////////////

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

// Obsolete.
float ReactionDiffusionSolverFVM::getMaxStableTimeStep() {
	if (!autoTimeSubStep) { return incTime; }
	else {
		std::vector<float> *fvMaxStableTimeSteps = new std::vector<float>(fieldDim.x*fieldDim.y*fieldDim.z);

		fvMaxStableTimeSteps->assign(fieldDim.x*fieldDim.y*fieldDim.z, incTime);
		unsigned int ind=0;


		#pragma omp parallel for shared (fieldDim) private (ind)
		for (ind=0;ind<fieldDim.x*fieldDim.y*fieldDim.z;ind++){
			fvMaxStableTimeSteps->at(ind) = this->getFieldFV(ind)->getMaxStableTimeStep();
		}
		return *min_element(fvMaxStableTimeSteps->begin(), fvMaxStableTimeSteps->end());
	}
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
	if (ind < 0 || ind > fieldDim.x*fieldDim.y*fieldDim.z) { throw BasicException("Point is not valid."); }
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

ReactionDiffusionSolverFV::ReactionDiffusionSolverFV(ReactionDiffusionSolverFVM *_solver, Point3D _coords, int _numFields) : solver(_solver), coords(_coords) {
	concentrationVecAux = std::vector<double>(_numFields, 0.0);
	concentrationVecOld = std::vector<double>(_numFields, 0.0);
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

void ReactionDiffusionSolverFV::solve() {
	std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr;
	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) {
		concentrationVecAux[i] = 0.0;
		for (fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
			std::vector<double> fluxVals = (this->*surfaceFluxFunctionPtrs[i][fv_itr->first])(i, fv_itr->first, fv_itr->second);
			concentrationVecAux[i] += fluxVals[0] * concentrationVecOld[i] + fluxVals[2];
			if (fv_itr->second != nullptr) { concentrationVecAux[i] += fluxVals[1] * fv_itr->second->getConcentrationOld(i); };
		}

		concentrationVecAux[i] += diagonalFunctionEval(i) * concentrationVecOld[i] + offDiagonalFunctionEval(i);
	}
}

double ReactionDiffusionSolverFV::solveStable() {
	double incTime = numeric_limits<double>::max();

	std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr;

	double den;

	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) {
		double den1 = 0.0;
		double den2 = 0.0;
		concentrationVecAux[i] = 0.0;

		for (fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
			std::vector<double> fluxVals = (this->*surfaceFluxFunctionPtrs[i][fv_itr->first])(i, fv_itr->first, fv_itr->second);

			den1 += fluxVals[0];
			concentrationVecAux[i] += fluxVals[2];

			if (fv_itr->second != nullptr) { 
				den2 += abs(fluxVals[1]);
				concentrationVecAux[i] += fluxVals[1] * fv_itr->second->getConcentrationOld(i);
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
	for (unsigned int fieldIndex = 0; fieldIndex < concentrationVecOld.size(); ++fieldIndex)
		concentrationVecOld[fieldIndex] = max(concentrationVecOld[fieldIndex] + concentrationVecAux[fieldIndex] * _incTime, 0.0);
	physTime += _incTime;
}

double ReactionDiffusionSolverFV::getMaxStableTimeStep() {
	double incTime = numeric_limits<double>::max();
	physTime = solver->getPhysicalTime();

	double den;

	std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr;

	for (unsigned int i = 0; i < concentrationVecOld.size(); ++i) {
		double den1 = 0.0;
		double den2 = 0.0;

		for (fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
			std::vector<double> fluxVals = (this->*surfaceFluxFunctionPtrs[i][fv_itr->first])(i, fv_itr->first, fv_itr->second);
			
			den1 += fluxVals[0];
			if (fv_itr->second != nullptr) { den2 += abs(fluxVals[1]); }
			
		}

		den = abs(den1 + diagonalFunctionEval(i)) + 1.001 * den2;

		if (den > 0) { incTime = min(incTime, (double) 1.0 / den); }

	}
	return incTime;
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

void ReactionDiffusionSolverFV::useDiffusiveSurfaces(unsigned int _fieldIndex) {
	for (std::map<unsigned int, ReactionDiffusionSolverFV *>::iterator fv_itr = neighborFVs.begin(); fv_itr != neighborFVs.end(); ++fv_itr) {
		surfaceFluxFunctionPtrs[_fieldIndex][fv_itr->first] = &ReactionDiffusionSolverFV::diffusiveSurfaceFlux;
	}
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
		CC3d_log(e.GetMsg());
		//cerr << e.GetMsg() << endl;
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}

void ReactionDiffusionSolverFV::setOffDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr) { 
	try { offDiagonalFunctions[_fieldIndex].SetExpr(_expr); }
	catch (mu::Parser::exception_type &e) {
		CC3d_log(e.GetMsg());
		//cerr << e.GetMsg() << endl;
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

double ReactionDiffusionSolverFV::getECMaterialDiffusivity(unsigned int _fieldIndex) {
	CellG *cell = solver->FVtoCellMap(this);
	if (cell == 0) { return solver->getECMaterialDiffusivity(_fieldIndex, coords); }
	else { return solver->getCellDiffusivityCoefficient(cell, _fieldIndex); }
}

double ReactionDiffusionSolverFV::getFieldDiffusivityField(unsigned int _fieldIndex) { return solver->getDiffusivityFieldPtVal(_fieldIndex, coords); }

double ReactionDiffusionSolverFV::getFieldDiffusivityInMedium(unsigned int _fieldIndex) {
	CellG * cell = solver->FVtoCellMap(this);
	if (cell == 0) { return getFieldDiffusivityField(_fieldIndex); }
	else { return solver->getCellDiffusivityCoefficient(cell, _fieldIndex); }
}

std::vector<double> ReactionDiffusionSolverFV::diffusiveSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) {
	if (_nFv == nullptr) {
		CC3d_log("Warning: diffusive surface flux for an unconnected FV pair!" );
		//cerr << "Warning: diffusive surface flux for an unconnected FV pair!" << endl;
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
		CC3d_log("Warning: permeable surface flux for an unconnected FV pair!");
		//cerr << "Warning: permeable surface flux for an unconnected FV pair!" << endl;
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

double ReactionDiffusionSolverFV::getFieldDiffusivity(unsigned int _fieldIndex) { return (this->*fieldDiffusivityFunctionPtrs[_fieldIndex])(_fieldIndex); }

///////////////////////////////////////////////////////////// Cell parameters/////////////////////////////////////////////////////////////
