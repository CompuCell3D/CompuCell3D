/*************************************************************************
*    CompuCell - A software framework for multimodel simulations of     *
* biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
*                             Indiana                                   *
*                                                                       *
* This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
*  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
*  CompuCell GNU General Public License RIDER you can redistribute it   *
* and/or modify it under the terms of the GNU General Public License as *
*  published by the Free Software Foundation; either version 2 of the   *
*         License, or (at your option) any later version.               *
*                                                                       *
* This program is distributed in the hope that it will be useful, but   *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

/**
@author T.J. Sego, Ph.D.
*/

#include "ECMaterialsSteppable.h"
#include "CompuCell3D/plugins/ECMaterials/ECMaterialsPlugin.h"

#include <ppl.h>
#include <chrono>
using namespace std::chrono;

ECMaterialsSteppable::ECMaterialsSteppable() : 
	cellFieldG(0),
	sim(0),
	potts(0),
	xmlData(0),
	boundaryStrategy(0),
	automaton(0),
	cellInventoryPtr(0),
	pUtils(0),
	numberOfMaterials(0),
	ECMaterialsInitialized(false),
	AnyInteractionsDefined(false),
	MaterialInteractionsDefined(false),
	FieldInteractionsDefined(false),
	CellInteractionsDefined(false)
{}

ECMaterialsSteppable::~ECMaterialsSteppable() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr = 0;
}

void ECMaterialsSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    potts = simulator->getPotts();
    cellInventoryPtr=& potts->getCellInventory();
    sim=simulator;

	pUtils = sim->getParallelUtils();
	lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);

	bool pluginAlreadyRegisteredFlag;

	// Get boundary pixel tracker plugin
	boundaryTrackerPlugin = (BoundaryPixelTrackerPlugin*)Simulator::pluginManager.get("BoundaryPixelTracker", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		CC3DXMLElement *BoundaryPixelTrackerXML = simulator->getCC3DModuleData("Plugin", "BoundaryPixelTracker");
		boundaryTrackerPlugin->init(simulator, BoundaryPixelTrackerXML);
	}

    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    fieldDim=cellFieldG->getDim();
    simulator->registerSteerableObject(this);

    update(_xmlData,true);

	pUtils->unsetLock(lockPtr);

}

void ECMaterialsSteppable::extraInit(Simulator *simulator){

    //PUT YOUR CODE HERE
}

void ECMaterialsSteppable::handleEvent(CC3DEvent & _event) {
	if (_event.id == LATTICE_RESIZE) {

	}
}

void ECMaterialsSteppable::start(){

    //PUT YOUR CODE HERE

}

void ECMaterialsSteppable::step(const unsigned int currentStep){

	if (!ECMaterialsInitialized) {
		pUtils->setLock(lockPtr);
		update(xmlData, true);
		pUtils->unsetLock(lockPtr);
	}

	if (!AnyInteractionsDefined) return;

    // Make a copy of the ECMaterialsField
	ECMaterialsField = ecMaterialsPlugin->getECMaterialField();
	Field3D<ECMaterialsData *> *ECMaterialsFieldOld = (Field3D<ECMaterialsData *>*) new WatchableField3D<ECMaterialsData *>(fieldDim, 0);
	ECMaterialsFieldOld = ECMaterialsField;

	bool MaterialInteractionsDefined = this->MaterialInteractionsDefined;
	bool FieldInteractionsDefined = this->FieldInteractionsDefined;

	boundaryStrategy = sim->getBoundaryStrategy()->getInstance();
	int nNeighbors = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

	// auto bmTimerStart = high_resolution_clock::now();

	if (MaterialInteractionsDefined || FieldInteractionsDefined) {
		concurrency::parallel_for(int(0), (int) fieldDim.z, [=](int z) {
			concurrency::parallel_for(int(0), (int) fieldDim.y, [=](int y) {
				concurrency::parallel_for(int(0), (int) fieldDim.x, [=](int x) {
					Point3D pt(x, y, z);
					const Point3D ptC = const_cast<Point3D&>(pt);
					int i, j;

					if (!cellFieldG->get(ptC)) {

						std::vector<float> qtyOld = ECMaterialsFieldOld->get(ptC)->getECMaterialsQuantityVec();
						std::vector<float> qtyNew = qtyOld;

						// Material interactions

						if (MaterialInteractionsDefined) {

							// Material reactions with at site
							if (MaterialReactionsDefined) {
								for each (i in idxMaterialReactionsDefined) {
									for each (j in idxidxMaterialReactionsDefined[i]) {

										qtyNew[i] += materialReactionCoefficientsByIndex[i][j][0] * qtyOld[j];

									}
								}
							}

							std::vector<float> qtyNeighbor;
							Point3D nPt;
							Neighbor neighbor;
							unsigned int nIdx;
							for (nIdx = 0; nIdx <= nNeighbors; ++nIdx) {
								neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
								if (!neighbor.distance) continue;
								nPt = neighbor.pt;

								const Point3D nPtC = const_cast<Point3D&>(nPt);
								if (cellFieldG->get(nPtC)) continue; // Detect extracellular sites

								qtyNeighbor = ECMaterialsFieldOld->get(nPtC)->getECMaterialsQuantityVec();

								// Material reactions with neighbors
								if (MaterialReactionsDefined) {
									for each (i in idxMaterialReactionsDefined) 
										for each (j in idxidxMaterialReactionsDefined[i]) {
											qtyNew[i] += materialReactionCoefficientsByIndex[i][j][1] * qtyNeighbor[j];
										}
								}

								// Material diffusion
								if (MaterialDiffusionDefined) {
									float materialDiffusionCoefficient;
									for each (i in idxMaterialDiffusionDefined) {
										materialDiffusionCoefficient = ECMaterialsVec->at(i).getMaterialDiffusionCoefficient();
										qtyNew[i] -= materialDiffusionCoefficient*qtyOld[i];
										qtyNew[i] += materialDiffusionCoefficient*qtyNeighbor[i];
									}
								}

							}

						}

						// Field interactions
						if (FieldInteractionsDefined) {
							// Update quantities
							for each (i in idxFromFieldInteractionsDefined) 
								for each (j in idxidxFromFieldInteractionsDefined[i]) {
									qtyNew[i] += fromFieldReactionCoefficientsByIndex[i][j] * fieldVec[j]->get(ptC);
								}

							// Generate field source terms
							// calculateMaterialToFieldInteractions(pt, qtyOld);
							float thisFieldVal;
							for each (i in idxToFieldInteractionsDefined) {
								thisFieldVal = fieldVec[i]->get(ptC);

								for each (j in idxidxToFieldInteractionsDefined[i]) {
									thisFieldVal += toFieldReactionCoefficientsByIndex[i][j] * qtyOld[j];
								}
								if (thisFieldVal < 0.0) thisFieldVal = 0.0;

								fieldVec[i]->set(ptC, thisFieldVal);

							}

						}

						ECMaterialsField->get(ptC)->setECMaterialsQuantityVec(checkQuantities(qtyNew));

					}

				});
			});
		});

	}

	// Cell interactions
	if (CellInteractionsDefined) { 
		
		calculateCellInteractions(ECMaterialsFieldOld);

	}

	// auto bmTimerStop = high_resolution_clock::now();
	// auto bmDuration = duration_cast<microseconds>(bmTimerStop - bmTimerStart);
	// cerr << "ECMaterialsSteppable time: " << bmDuration.count() << " ms" << endl;

}

void ECMaterialsSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	// Do this to enable initializations independently of startup routine
	if (ECMaterialsInitialized)
		return;

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton);

	// Get ECMaterials plugin
	bool pluginAlreadyRegisteredFlag;
	ecMaterialsPlugin = (ECMaterialsPlugin*)Simulator::pluginManager.get("ECMaterials", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) {
		return;
	}

	// Forward steppable to plugin
	ecMaterialsPlugin->setECMaterialsSteppable(this);

	// Get cell types
	cellTypeNames.clear();
	std::string cellTypeName;
	for (int j = 0; j <= (int)automaton->getMaxTypeId(); ++j) {
		
		cellTypeName = automaton->getTypeName(j);

		cerr << "   Got type " << cellTypeName << ": " << j << endl;

		cellTypeNames.insert(make_pair(automaton->getTypeName(j), j));
	}
	numberOfCellTypes = cellTypeNames.size();

	// Initialize preliminaries from ECMaterials plugin
	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();
	ECMaterialsField = ecMaterialsPlugin->getECMaterialField();
	ECMaterialsVec = ecMaterialsPlugin->getECMaterialsVecPtr();

	// Initialize preallocation and performance variables
	std::vector<bool> hasMaterialReactionsDefined = std::vector<bool>(numberOfMaterials, false);
	std::vector<bool> hasMaterialDiffusionDefined = std::vector<bool>(numberOfMaterials, false);
	std::vector<bool> hasToFieldInteractionsDefined;
	std::vector<bool> hasFromFieldInteractionsDefined = std::vector<bool>(numberOfMaterials, false);
	std::vector<bool> hasCellInteractionsDefined = std::vector<bool>(numberOfMaterials, false);

	idxMaterialReactionsDefined.clear();
	idxidxMaterialReactionsDefined.clear();
	idxidxMaterialReactionsDefined.resize(numberOfMaterials);
	idxMaterialDiffusionDefined.clear();
	idxToFieldInteractionsDefined.clear();
	idxidxToFieldInteractionsDefined.clear();
	idxFromFieldInteractionsDefined.clear();
	idxidxFromFieldInteractionsDefined.clear();
	idxidxFromFieldInteractionsDefined.resize(numberOfMaterials);
	idxCellInteractionsDefined.clear();

    // Gather XML user specifications
	CC3DXMLElementList MaterialInteractionXMLVec = xmlData->getElements("MaterialInteraction");
	CC3DXMLElementList FieldInteractionXMLVec = xmlData->getElements("FieldInteraction");
	CC3DXMLElementList MaterialDiffusionXMLVec = xmlData->getElements("MaterialDiffusion");
	CC3DXMLElementList CellInteractionXMLVec = xmlData->getElements("CellInteraction");

	std::string thisMaterialName;
	std::string catalystName;
	float coeff;
	
	MaterialInteractionsDefined = false;

	int materialIndex;
	int catalystIndex;
	if (xmlData->findElement("MaterialInteraction")) {
		MaterialReactionsDefined = true;

		cerr << "Getting ECMaterial interactions..." << endl;

		for (int XMLidx = 0; XMLidx < MaterialInteractionXMLVec.size(); ++XMLidx) {

			thisMaterialName = MaterialInteractionXMLVec[XMLidx]->getAttribute("ECMaterial");

			cerr << "   ECMaterial " << thisMaterialName << endl;

			catalystName = MaterialInteractionXMLVec[XMLidx]->getFirstElement("Catalyst")->getText();
			cerr << "   Catalyst " << catalystName << endl;

			coeff = (float) MaterialInteractionXMLVec[XMLidx]->getFirstElement("ConstantCoefficient")->getDouble();
			cerr << "   Constant coefficient " << coeff << endl;

			int interactionOrder = 0;
			if (MaterialInteractionXMLVec[XMLidx]->findElement("NeighborhoodOrder")) {
				interactionOrder = MaterialInteractionXMLVec[XMLidx]->getFirstElement("NeighborhoodOrder")->getInt();
			}
			
			materialIndex = getECMaterialIndexByName(thisMaterialName);

			ASSERT_OR_THROW("ECMaterial not defined in ECMaterials plugin.", materialIndex >= 0);

			catalystIndex = getECMaterialIndexByName(catalystName);

			ASSERT_OR_THROW("Catalyst not defined in ECMaterials plugin.", catalystIndex >= 0);

			cerr << "   ";
			ECMaterialsVec->at(materialIndex).setMaterialReactionCoefficientByName(catalystName, coeff, interactionOrder);
			hasMaterialReactionsDefined[materialIndex] = true;

			idxidxMaterialReactionsDefined[materialIndex].push_back(catalystIndex);

		}

		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) if (hasMaterialReactionsDefined[mtlIdx]) idxMaterialReactionsDefined.push_back(mtlIdx); 

	}
	else MaterialReactionsDefined = false;

	if (xmlData->findElement("MaterialDiffusion")) {
		MaterialDiffusionDefined = true;

		cerr << "Getting ECMaterial diffusion..." << endl;

		for (int XMLidx = 0; XMLidx < MaterialDiffusionXMLVec.size(); ++XMLidx) {

			thisMaterialName = MaterialDiffusionXMLVec[XMLidx]->getAttribute("ECMaterial");

			cerr << "   ECMaterial " << thisMaterialName << endl;

			catalystName = thisMaterialName;

			coeff = (float)MaterialDiffusionXMLVec[XMLidx]->getDouble();
			cerr << "      Diffusion coefficient " << coeff << endl;

			ASSERT_OR_THROW("Invalid diffusion coefficient: must be positive.", coeff >= 0);

			materialIndex = getECMaterialIndexByName(thisMaterialName);

			ASSERT_OR_THROW("ECMaterial not defined in ECMaterials plugin.", materialIndex >= 0);

			ECMaterialsVec->at(materialIndex).setMaterialDiffusionCoefficient(coeff);
			hasMaterialDiffusionDefined[materialIndex] = true;

		}
		
		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) if (hasMaterialDiffusionDefined[mtlIdx]) idxMaterialDiffusionDefined.push_back(mtlIdx);

	}
	else MaterialDiffusionDefined = false;

	MaterialInteractionsDefined = MaterialReactionsDefined || MaterialDiffusionDefined;

	if (MaterialInteractionsDefined) constructMaterialReactionCoefficients();

	if (xmlData->findElement("FieldInteraction")) {
		FieldInteractionsDefined = true;

		cerr << "Getting ECMaterial field interactions..." << endl;

		map<string, Field3D<float>*> & nameFieldMap = sim->getConcentrationFieldNameMap();

		fieldVec.clear();
		fieldNames.clear();
		for (std::map<string, Field3D<float>*>::iterator itr = nameFieldMap.begin(); itr != nameFieldMap.end(); ++itr) {
			fieldNames.insert(make_pair(itr->first, fieldNames.size()));
			fieldVec.push_back(itr->second);
		}
		numberOfFields = fieldVec.size();
		hasToFieldInteractionsDefined.assign(numberOfFields, false);
		idxidxToFieldInteractionsDefined.resize(numberOfFields);

		std::string reactantName;
		std::string productName;
		bool FieldIsDefined;
		bool toField;
		std::string thisFieldName;
		int fieldIndex;
		for (int XMLidx = 0; XMLidx < FieldInteractionXMLVec.size(); ++XMLidx) {

			cerr << "   Getting field interaction " << XMLidx + 1 << endl;

			ASSERT_OR_THROW("A reactant must be defined (using Reactant)", FieldInteractionXMLVec[XMLidx]->findElement("Reactant"));
			reactantName = FieldInteractionXMLVec[XMLidx]->getFirstElement("Reactant")->getText();
			cerr << "      Reactant " << reactantName << endl;

			ASSERT_OR_THROW("A catalyst must be defined (using Catalyst)", FieldInteractionXMLVec[XMLidx]->findElement("Catalyst"));
			catalystName = FieldInteractionXMLVec[XMLidx]->getFirstElement("Catalyst")->getText();
			cerr << "      Catalyst " << catalystName << endl;

			ASSERT_OR_THROW("A constant coefficient must be defined (using ConstantCoefficient)", FieldInteractionXMLVec[XMLidx]->findElement("ConstantCoefficient"));
			coeff = (float)FieldInteractionXMLVec[XMLidx]->getFirstElement("ConstantCoefficient")->getDouble();
			cerr << "      Constant coefficient " << coeff << endl;

			FieldIsDefined = false;

			// get field
			//		when generating field source terms
			fieldIndex = getFieldIndexByName(reactantName);
			if (fieldIndex >= 0) {
				FieldIsDefined = true;
				toField = true;
				thisFieldName = reactantName;
				thisMaterialName = catalystName;
			}
			//		when generating material source terms
			if (getFieldIndexByName(catalystName) >= 0) {

				ASSERT_OR_THROW("Cannot define field-field interactions here.", !FieldIsDefined);

				fieldIndex = getFieldIndexByName(catalystName);
				FieldIsDefined = true;
				toField = false;
				thisFieldName = catalystName;
				thisMaterialName = reactantName;
			}

			ASSERT_OR_THROW("Field " + thisFieldName + " not registered.", FieldIsDefined);

			// get material
			materialIndex = getECMaterialIndexByName(thisMaterialName);

			ASSERT_OR_THROW("ECMaterial " + thisMaterialName + " not registered in ECMaterials plugin.", materialIndex >= 0);

			// load valid info

			if (toField) {
				cerr << "      Mapping ECMaterial onto field" << endl;
				ECMaterialsVec->at(materialIndex).setToFieldReactionCoefficientByName(thisFieldName, coeff);
				hasToFieldInteractionsDefined[fieldIndex] = true;
				idxidxToFieldInteractionsDefined[fieldIndex].push_back(materialIndex);
			}
			else {
				cerr << "      Mapping field onto ECMaterial" << endl;
				ECMaterialsVec->at(materialIndex).setFromFieldReactionCoefficientByName(thisFieldName, coeff);
				hasFromFieldInteractionsDefined[materialIndex] = true;
				idxidxFromFieldInteractionsDefined[materialIndex].push_back(fieldIndex);
			}

		}

		for (int fldIdx = 0; fldIdx < numberOfFields; ++fldIdx) {
			if (hasToFieldInteractionsDefined[fldIdx]) idxToFieldInteractionsDefined.push_back(fldIdx);
		}
		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) {
			if (hasFromFieldInteractionsDefined[mtlIdx]) idxFromFieldInteractionsDefined.push_back(mtlIdx);
		}

		constructFieldReactionCoefficients();

	}

	if (xmlData->findElement("CellInteraction")) {
		CellInteractionsDefined = true;

		std::string methodName;
		std::string thisCellType;
		std::string thisCellTypeNew;
		int cellTypeIndex;
		int cellTypeNewIndex;
		for (int XMLidx = 0; XMLidx < CellInteractionXMLVec.size(); ++XMLidx) {
			thisCellTypeNew.clear();

			cerr << "   Getting cell interaction " << XMLidx + 1 << endl;

			ASSERT_OR_THROW("A method must be defined for each cell interaction (using Method).", CellInteractionXMLVec[XMLidx]->findAttribute("Method"));
			methodName = CellInteractionXMLVec[XMLidx]->getAttribute("Method");
			cerr << "      Method " << methodName << endl;

			ASSERT_OR_THROW("An ECMaterial must be defined for each cell interaction (using ECMaterial).", CellInteractionXMLVec[XMLidx]->findElement("ECMaterial"));
			thisMaterialName = CellInteractionXMLVec[XMLidx]->getFirstElement("ECMaterial")->getText();
			cerr << "      ECMaterial " << thisMaterialName << endl;

			ASSERT_OR_THROW("A probability must be defined for each cell interaction (using Probability).", CellInteractionXMLVec[XMLidx]->findElement("Probability"));
			coeff = (float) CellInteractionXMLVec[XMLidx]->getFirstElement("Probability")->getDouble();
			cerr << "      Probability " << coeff << endl;

			ASSERT_OR_THROW("A cell type must be defined for each cell interaction (using CellType).", CellInteractionXMLVec[XMLidx]->findElement("CellType"));
			thisCellType = CellInteractionXMLVec[XMLidx]->getFirstElement("CellType")->getText();
			cerr << "      Cell type " << thisCellType << endl;

			if (CellInteractionXMLVec[XMLidx]->findElement("CellTypeNew")) {
				thisCellTypeNew = CellInteractionXMLVec[XMLidx]->getFirstElement("CellTypeNew")->getText();
				cerr << "      New cell type " << thisCellTypeNew << endl;
			}

			if (methodName == "Differentiation") {
				ASSERT_OR_THROW("A new cell type must be defined for each cell differentiation interaction (using CellTypeNew).", thisCellTypeNew.length() > 0);
			}

			// perform checks
			materialIndex = getECMaterialIndexByName(thisMaterialName);
			ASSERT_OR_THROW("ECMaterial " + thisMaterialName + " not registered.", materialIndex >= 0);
			ASSERT_OR_THROW("Probability must be non-negative", coeff >= 0);
			cellTypeIndex = getCellTypeIndexByName(thisCellType);
			ASSERT_OR_THROW("Cell type " + thisCellType + " not registered.", cellTypeIndex >= 0);
			cellTypeNewIndex = getCellTypeIndexByName(thisCellTypeNew);
			if (methodName == "Differentiation") {
				ASSERT_OR_THROW("New cell type " + thisCellTypeNew + " not registered.", cellTypeNewIndex >= 0);
			}

			if (methodName == "Proliferation") {
				ECMaterialsVec->at(materialIndex).setCellTypeCoefficientsProliferation(thisCellType, coeff, thisCellTypeNew);
			}
			else if (methodName == "Differentiation") {
				ECMaterialsVec->at(materialIndex).setCellTypeCoefficientsDifferentiation(thisCellType, coeff, thisCellTypeNew);
			}
			else if (methodName == "Death") {
				ECMaterialsVec->at(materialIndex).setCellTypeCoefficientsDeath(thisCellType, coeff);
			}
			else ASSERT_OR_THROW("Undefined cell response method: use Proliferation, Differentiation, or Death.", false);

			hasCellInteractionsDefined[materialIndex] = true;

		}

		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) if (hasCellInteractionsDefined[mtlIdx]) idxCellInteractionsDefined.push_back(mtlIdx);

		constructCellTypeCoefficients();
	}

	ECMaterialsInitialized = true;
	AnyInteractionsDefined = MaterialInteractionsDefined || FieldInteractionsDefined || CellInteractionsDefined;
}

void ECMaterialsSteppable::calculateCellInteractions(Field3D<ECMaterialsData *> *ECMaterialsFieldOld) {

	ASSERT_OR_THROW("ECMaterials steppable not yet initialized.", ECMaterialsInitialized);

	ecMaterialsPlugin->resetCellResponse();
	
	CellG * cell = 0;

	CellInventory::cellInventoryIterator cInvItr;
	CellInventory * cellInventoryPtr = &potts->getCellInventory();
	int nNeighbors = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

	// Initialize information for counting cell interactions
	int numberOfCells = cellInventoryPtr->getSize();
	cellResponses.clear();

	float probAccumProlif;
	std::vector<float> probAccumProlifAsym;
	std::vector<float> probAccumDiff;
	float probAccumDeath;
	
	//		randomly order evaluation of responses
	std::vector<std::vector<int> > responseOrderSet = permutations(4);
	std::vector<int> responseOrder = responseOrderSet[rand() % responseOrderSet.size()];
	std::tuple<float, float, std::vector<float>, std::vector<float> > probs;

	// Loop over cells and consider each response
	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
		cell = cellInventoryPtr->getCell(cInvItr);

		probs = calculateCellProbabilities(cell, ECMaterialsFieldOld);

		probAccumProlif = std::get<0>(probs);
		probAccumDeath = std::get<1>(probs);
		probAccumDiff = std::get<2>(probs);
		probAccumProlifAsym = std::get<3>(probs);

		// Consider each response in random order of response
		// Maybe move this out and parallilize before this block
		bool resp = false;
		std::string Action;
		std::string CellTypeDiff = "";
		for each (int respIdx in responseOrder) {
			switch (respIdx) {
			case 0 : // proliferation
				resp = testProb(probAccumProlif);
				Action = "Proliferation";
				break;
			case 1 : // death
				resp = testProb(probAccumDeath);
				Action = "Death";
				break;
			case 2 : // differentiation
				Action = "Differentiation";
				for (std::map<std::string, int>::iterator mitr = cellTypeNames.begin(); mitr != cellTypeNames.end(); ++mitr) {
					resp = testProb(probAccumDiff[mitr->second]);
					CellTypeDiff = mitr->first;
					if (resp) break;
				}
				break;
			case 3 : // asymmetric division
				Action = "Proliferation";
				for (std::map<std::string, int>::iterator mitr = cellTypeNames.begin(); mitr != cellTypeNames.end(); ++mitr) {
					resp = testProb(probAccumProlifAsym[mitr->second]);
					CellTypeDiff = mitr->first;
					if (resp) break;
				}
				break;
			}
			if (resp) {
				ecMaterialsPlugin->addCellResponse(cell, Action, CellTypeDiff);
				break;
			}
		}

	}

}

std::tuple<float, float, std::vector<float>, std::vector<float> > ECMaterialsSteppable::calculateCellProbabilities(CellG *cell, Field3D<ECMaterialsData *> *_ecmaterialsField, std::vector<bool> _resultsSelect) {
	if (!_ecmaterialsField) _ecmaterialsField = ECMaterialsField;

	float probAccumProlif = 0.0;
	std::vector<float> probAccumProlifAsym(numberOfCellTypes, 0.0);
	std::vector<float> probAccumDiff(numberOfCellTypes, 0.0);
	float probAccumDeath = 0.0;

	if (!cell) return make_tuple(probAccumProlif, probAccumDeath, probAccumDiff, probAccumProlifAsym);

	Neighbor neighbor;
	CellG *nCell = 0;
	Point3D pt;
	int cellId = (int)cell->id;
	int cellType = (int)cell->type;
	std::vector<float> qtyOld(numberOfMaterials, 0.0);

	// Loop over cell boundaries to find interfaces with ECMaterials
	std::set<BoundaryPixelTrackerData > *boundarySet = boundaryTrackerPlugin->getPixelSetForNeighborOrderPtr(cell, neighborOrder);
	for (std::set<BoundaryPixelTrackerData >::iterator bInvItr = boundarySet->begin(); bInvItr != boundarySet->end(); ++bInvItr) {

		// Loop over cell boundary neighbors and accumulate probabilities from medium sites
		pt = bInvItr->pixel;

		for (unsigned int nIdx = 0; nIdx <= nNeighbors; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

			if (!neighbor.distance) continue;

			nCell = cellFieldG->get(neighbor.pt);

			if (nCell) continue;

			qtyOld = _ecmaterialsField->get(neighbor.pt)->getECMaterialsQuantityVec();

			for each (int i in idxCellInteractionsDefined) {
				if (_resultsSelect[0]) probAccumProlif += CellTypeCoefficientsProliferationByIndex[i][cellType] * qtyOld[i];
				if (_resultsSelect[1]) probAccumDeath += CellTypeCoefficientsDeathByIndex[i][cellType] * qtyOld[i];
				if (_resultsSelect[2] && _resultsSelect[3]) {
					for (int typeIdx = 0; typeIdx < probAccumDiff.size(); ++typeIdx) {
						probAccumProlifAsym[typeIdx] += CellTypeCoefficientsProliferationAsymByIndex[i][cellType][typeIdx] * qtyOld[i];
						probAccumDiff[typeIdx] += CellTypeCoefficientsDifferentiationByIndex[i][cellType][typeIdx] * qtyOld[i];
					}
				}
				else if (_resultsSelect[2]) for (int typeIdx = 0; typeIdx < probAccumDiff.size(); ++typeIdx) probAccumDiff[typeIdx] += CellTypeCoefficientsDifferentiationByIndex[i][cellType][typeIdx] * qtyOld[i];
				else if (_resultsSelect[3]) for (int typeIdx = 0; typeIdx < probAccumDiff.size(); ++typeIdx) probAccumProlifAsym[typeIdx] += CellTypeCoefficientsProliferationAsymByIndex[i][cellType][typeIdx] * qtyOld[i];
			}

		}

	}

	return make_tuple(probAccumProlif, probAccumDeath, probAccumDiff, probAccumProlifAsym);
}

float ECMaterialsSteppable::calculateCellProbabilityProliferation(CellG *cell, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	return std::get<0>(calculateCellProbabilities(cell, _ecmaterialsField, { true, false, false, false }));
}

float ECMaterialsSteppable::calculateCellProbabilityDeath(CellG *cell, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	return std::get<1>(calculateCellProbabilities(cell, _ecmaterialsField, { false, true, false, false }));
}

float ECMaterialsSteppable::calculateCellProbabilityDifferentiation(CellG *cell, std::string newCellType, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	int newCellTypeIndex = getCellTypeIndexByName(newCellType);
	if (newCellTypeIndex < 0) return 0.0;
	return std::get<2>(calculateCellProbabilities(cell, _ecmaterialsField, { false, false, true, false }))[newCellTypeIndex];
}

float ECMaterialsSteppable::calculateCellProbabilityAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	int newCellTypeIndex = getCellTypeIndexByName(newCellType);
	if (newCellTypeIndex < 0) return 0.0;
	return std::get<3>(calculateCellProbabilities(cell, _ecmaterialsField, { false, false, false, true }))[newCellTypeIndex];
}

bool ECMaterialsSteppable::getCellResponseProliferation(CellG *cell, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	return testProb(calculateCellProbabilityProliferation(cell, _ecmaterialsField));
}
bool ECMaterialsSteppable::getCellResponseDeath(CellG *cell, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	return testProb(calculateCellProbabilityDeath(cell, _ecmaterialsField));
}
bool ECMaterialsSteppable::getCellResponseDifferentiation(CellG *cell, std::string newCellType, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	return testProb(calculateCellProbabilityDifferentiation(cell, newCellType, _ecmaterialsField));
}
bool ECMaterialsSteppable::getCellResponseAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<ECMaterialsData *> *_ecmaterialsField) {
	return testProb(calculateCellProbabilityAsymmetricDivision(cell, newCellType, _ecmaterialsField));
}

void ECMaterialsSteppable::constructFieldReactionCoefficients() {

	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();
	numberOfFields = fieldVec.size();

	toFieldReactionCoefficientsByIndex = std::vector<std::vector<float> >(numberOfFields, std::vector<float>(numberOfMaterials, 0.0));
	fromFieldReactionCoefficientsByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfFields, 0.0));

	for (std::vector<int>::iterator i = idxFromFieldInteractionsDefined.begin(); i != idxFromFieldInteractionsDefined.end(); ++i) {
		for (std::vector<int>::iterator j = idxidxFromFieldInteractionsDefined[*i].begin(); j != idxidxFromFieldInteractionsDefined[*i].end(); ++j) {
			fromFieldReactionCoefficientsByIndex[*i][*j] = ECMaterialsVec->at(*i).getFromFieldReactionCoefficientByName(getFieldNameByIndex(*j));
		}
	}
	for (std::vector<int>::iterator i = idxToFieldInteractionsDefined.begin(); i != idxToFieldInteractionsDefined.end(); ++i) {
		for (std::vector<int>::iterator j = idxidxToFieldInteractionsDefined[*i].begin(); j != idxidxToFieldInteractionsDefined[*i].end(); ++j) {
			toFieldReactionCoefficientsByIndex[*i][*j] = ECMaterialsVec->at(*j).getToFieldReactionCoefficientByName(getFieldNameByIndex(*i));
		}
	}

}

void ECMaterialsSteppable::constructMaterialReactionCoefficients() {

	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();

	materialReactionCoefficientsByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(2, 0.0)));

	for (std::vector<int>::iterator i = idxMaterialReactionsDefined.begin(); i != idxMaterialReactionsDefined.end(); ++i) {
		for (std::vector<int>::iterator j = idxidxMaterialReactionsDefined[*i].begin(); j != idxidxMaterialReactionsDefined[*i].end(); ++j) {
			materialReactionCoefficientsByIndex[*i][*j] = ECMaterialsVec->at(*i).getMaterialReactionCoefficientsByName(ECMaterialsVec->at(*j).getName());
		}
	}
}

void ECMaterialsSteppable::constructCellTypeCoefficients() {

	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsProliferationByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));
	CellTypeCoefficientsProliferationAsymByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfCellTypes, std::vector<float>(numberOfCellTypes, 0.0)));
	CellTypeCoefficientsDifferentiationByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfCellTypes, std::vector<float>(numberOfCellTypes, 0.0)));
	CellTypeCoefficientsDeathByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));

	for each (int i in idxCellInteractionsDefined) {
		for (int j = 0; j < CellTypeCoefficientsProliferationByIndex[i].size(); ++j) {

			std::string cellTypeName = automaton->getTypeName(j);

			CellTypeCoefficientsProliferationByIndex[i][j] = ECMaterialsVec->at(i).getCellTypeCoefficientsProliferation(cellTypeName);
			CellTypeCoefficientsDeathByIndex[i][j] = ECMaterialsVec->at(i).getCellTypeCoefficientsDeath(cellTypeName);
			for (int k = 0; k < CellTypeCoefficientsDifferentiationByIndex[i][j].size(); ++k) {
				CellTypeCoefficientsProliferationAsymByIndex[i][j][k] = ECMaterialsVec->at(i).getCellTypeCoefficientsProliferationAsymmetric(cellTypeName, automaton->getTypeName(k));
				CellTypeCoefficientsDifferentiationByIndex[i][j][k] = ECMaterialsVec->at(i).getCellTypeCoefficientsDifferentiation(cellTypeName, automaton->getTypeName(k));
			}
		}
	}

}

void ECMaterialsSteppable::constructCellTypeCoefficientsProliferation() {

	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsProliferationByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));

	for (std::vector<int>::iterator i = idxCellInteractionsDefined.begin(); i != idxCellInteractionsDefined.end(); ++i) {
		for (int j = 0; j < numberOfCellTypes; ++j) {
			CellTypeCoefficientsProliferationByIndex[*i][j] = ECMaterialsVec->at(*i).getCellTypeCoefficientsProliferation(automaton->getTypeName(j));
		}
	}

}

void ECMaterialsSteppable::constructCellTypeCoefficientsDifferentiation() {

	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsDifferentiationByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfCellTypes, std::vector<float>(numberOfCellTypes, 0.0)));

	for (std::vector<int>::iterator i = idxCellInteractionsDefined.begin(); i != idxCellInteractionsDefined.end(); ++i) {
		for (int j = 0; j < numberOfCellTypes; ++j) {
			for (int k = 0; k < numberOfCellTypes; ++k) {
				CellTypeCoefficientsDifferentiationByIndex[*i][j][k] = ECMaterialsVec->at(*i).getCellTypeCoefficientsDifferentiation(automaton->getTypeName(j), automaton->getTypeName(k));
			}
		}
	}

}

void ECMaterialsSteppable::constructCellTypeCoefficientsDeath() {

	numberOfMaterials = ecMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsDeathByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));

	for (std::vector<int>::iterator i = idxCellInteractionsDefined.begin(); i != idxCellInteractionsDefined.end(); ++i) {
		for (int j = 0; j < numberOfCellTypes; ++j) {
			CellTypeCoefficientsDeathByIndex[*i][j] = ECMaterialsVec->at(*i).getCellTypeCoefficientsDeath(automaton->getTypeName(j));
		}
	}

}

void ECMaterialsSteppable::calculateMaterialToFieldInteractions(const Point3D &pt, std::vector<float> _qtyOld) {
	float thisFieldVal;
	for (std::vector<int>::iterator i = idxToFieldInteractionsDefined.begin(); i != idxToFieldInteractionsDefined.end(); ++i) {
		thisFieldVal = fieldVec[*i]->get(pt);
		for (std::vector<int>::iterator j = idxidxToFieldInteractionsDefined[*i].begin(); j != idxidxToFieldInteractionsDefined[*i].end(); ++j) {
			thisFieldVal += toFieldReactionCoefficientsByIndex[*i][*j] * _qtyOld[*j];
		}
		if (thisFieldVal < 0.0) thisFieldVal = 0.0;
		fieldVec[*i]->set(pt, thisFieldVal);
	}
}

std::vector<std::vector<int> > ECMaterialsSteppable::permutations(int _numberOfVals) {
	std::vector<std::vector<int> > permutationVec;
	std::vector<int> permutationSet(_numberOfVals, 0);
	for (int i = 0; i < _numberOfVals; ++i) permutationSet[i] = i;
	do {
		permutationVec.push_back(permutationSet);
	} while (next_permutation(permutationSet.begin(), permutationSet.end()));
	return permutationVec;
}

std::string ECMaterialsSteppable::toString(){
   return "ECMaterialsSteppable";
}

std::string ECMaterialsSteppable::steerableName(){
   return toString();
}