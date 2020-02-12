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

#include "NCMaterialsSteppable.h"
#include "CompuCell3D/plugins/NCMaterials/NCMaterialsPlugin.h"

#include <ppl.h>
#include <concurrent_vector.h>
#include <muParser/muParser.h>

NCMaterialsExpressionEvaluator::NCMaterialsExpressionEvaluator(unsigned int _numMaterials, unsigned int _numFields) : 
	step(0.0), 
	numMaterials(_numMaterials),
	numFields(_numFields)
{}

void NCMaterialsExpressionEvaluator::init(Point3D _pt) {
	qty = std::vector<double>(numMaterials, 0.0);
	fld = std::vector<double>(numFields, 0.0);

	exprQuantities = std::vector<mu::Parser>(numMaterials, templateMuParserFunction(_pt));
	exprFields = std::vector<mu::Parser>(numFields, templateMuParserFunction(_pt));
}

mu::Parser NCMaterialsExpressionEvaluator::templateMuParserFunction(Point3D _pt) {
	mu::Parser _fcn = mu::Parser();
	_fcn.SetExpr("0.0");
	_fcn.DefineConst("x", (double)_pt.x);
	_fcn.DefineConst("y", (double)_pt.y);
	_fcn.DefineConst("z", (double)_pt.z);
	_fcn.DefineVar("t", &step);
	return _fcn;
}

void NCMaterialsExpressionEvaluator::setSymbol(std::string _sym, double *_val) {
	for (unsigned int i = 0; i < numMaterials; ++i) exprQuantities[i].DefineVar(_sym, _val);
	for (unsigned int i = 0; i < numFields; ++i) exprFields[i].DefineVar(_sym, _val);
}

void NCMaterialsExpressionEvaluator::setSymbols(std::vector<string> _materialSymbols, std::vector<string> _fieldSymbols) {
	for (unsigned int i = 0; i < numMaterials; ++i)
		setSymbol(_materialSymbols[i], &qty[i]);

	for (unsigned int i = 0; i < numFields; ++i)
		setSymbol(_fieldSymbols[i], &fld[i]);
}

void NCMaterialsExpressionEvaluator::setMaterialExpression(unsigned int _materialIndex, std::string _expr) {
	try { exprQuantities[_materialIndex].SetExpr(_expr); }
	catch (mu::Parser::exception_type &e) {
		cerr << e.GetMsg() << endl;
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}

void NCMaterialsExpressionEvaluator::setFieldExpression(unsigned int _fieldIndex, std::string _expr) {
	try { exprFields[_fieldIndex].SetExpr(_expr); }
	catch (mu::Parser::exception_type &e) {
		cerr << e.GetMsg() << endl;
		ASSERT_OR_THROW(e.GetMsg(), 0);
	}
}

void NCMaterialsExpressionEvaluator::updateInternals(std::vector<float> _qty, std::vector<float> _fld) {

	for (unsigned int i = 0; i < qty.size(); ++i)
		qty[i] = (double)_qty[i];

	for (unsigned int i = 0; i < fld.size(); ++i) 
		fld[i] = (double)_fld[i];

}

std::vector<float> NCMaterialsExpressionEvaluator::evalMaterialExpressions(double _step, std::vector<float> _qty, std::vector<float> _fld) {
	step = _step;

	updateInternals(_qty, _fld);

	std::vector<float> y = std::vector<float>(numMaterials, 0.0);
	for (unsigned int i = 0; i < numMaterials; ++i) y[i] = static_cast<float>(exprQuantities[i].Eval());
	return y;
}

std::vector<float> NCMaterialsExpressionEvaluator::evalFieldExpressions(double _step, std::vector<float> _qty, std::vector<float> _fld) {
	step = _step;

	updateInternals(_qty, _fld);

	std::vector<float> y = std::vector<float>(numFields, 0.0);
	for (unsigned int i = 0; i < numFields; ++i) y[i] = static_cast<float>(exprFields[i].Eval());
	return y;
}

std::tuple<std::vector<float>, std::vector<float> > NCMaterialsExpressionEvaluator::evalExpressions(double _step, std::vector<float> _qty, std::vector<float> _fld) {
	step = _step;

	updateInternals(_qty, _fld);

	std::vector<float> yMtl = std::vector<float>(numMaterials, 0.0);
	std::vector<float> yFld = std::vector<float>(numFields, 0.0);
	for (unsigned int i = 0; i < numMaterials; ++i) {
		try { yMtl[i] = static_cast<float>(exprQuantities[i].Eval()); }
		catch (mu::Parser::exception_type &e) {
			cerr << "Message:  " << e.GetMsg() << "\n";
			cerr << "Formula:  " << e.GetExpr() << "\n";
			cerr << "Token:    " << e.GetToken() << "\n";
			cerr << "Position: " << e.GetPos() << "\n";
			cerr << "Errc:     " << e.GetCode() << "\n";
		}
	}
	for (unsigned int i = 0; i < numFields; ++i) {
		try { yFld[i] = static_cast<float>(exprFields[i].Eval()); }
		catch (mu::Parser::exception_type &e) {
			cerr << "Message:  " << e.GetMsg() << "\n";
			cerr << "Formula:  " << e.GetExpr() << "\n";
			cerr << "Token:    " << e.GetToken() << "\n";
			cerr << "Position: " << e.GetPos() << "\n";
			cerr << "Errc:     " << e.GetCode() << "\n";
		}
	}

	return make_tuple(yMtl, yFld);
}

NCMaterialsSteppable::NCMaterialsSteppable() : 
	cellFieldG(0),
	sim(0),
	potts(0),
	xmlData(0),
	boundaryStrategy(0),
	automaton(0),
	cellInventoryPtr(0),
	pUtils(0),
	numberOfMaterials(0), 
	neighborOrder(1),
	NCMaterialsInitialized(false),
	AnyInteractionsDefined(false),
	MaterialInteractionsDefined(false),
	FieldInteractionsDefined(false), 
	GeneralInteractionsDefined(false),
	CellInteractionsDefined(false)
{}

NCMaterialsSteppable::~NCMaterialsSteppable() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr = 0;

	if (GeneralInteractionsDefined) for (unsigned int i = 0; i < NCMExprEvalLocal.size(); ++i) delete NCMExprEvalLocal[i];
}

void NCMaterialsSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
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
    fieldDim = cellFieldG->getDim();
    simulator->registerSteerableObject(this);

    update(_xmlData,true);

	pUtils->unsetLock(lockPtr);

}

void NCMaterialsSteppable::handleEvent(CC3DEvent & _event) {
	if (_event.id == LATTICE_RESIZE) {

	}
}

void NCMaterialsSteppable::step(const unsigned int currentStep){

	if (!NCMaterialsInitialized) {
		pUtils->setLock(lockPtr);
		update(xmlData, true);
		pUtils->unsetLock(lockPtr);
	}

	if (!AnyInteractionsDefined) return;

    // Copy old field quantities
	NCMaterialsField = ncMaterialsPlugin->getNCMaterialField();
	concurrency::parallel_for(int(0), (int)(fieldDim.x*fieldDim.y*fieldDim.z), [&](int ind) {
		NCMaterialsField->get(this->ind2pt(ind))->setNCMaterialsQuantityVecOld(); });

	bool MaterialInteractionsDefined = this->MaterialInteractionsDefined;
	bool FieldInteractionsDefined = this->FieldInteractionsDefined;
	bool GeneralInteractionsDefined = this->GeneralInteractionsDefined;

	boundaryStrategy = sim->getBoundaryStrategy()->getInstance();
	nNeighbors = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
	int nNeighbors = this->nNeighbors;

	if (MaterialInteractionsDefined || FieldInteractionsDefined || GeneralInteractionsDefined) {

		concurrency::parallel_for(int(0), (int)(fieldDim.x*fieldDim.y*fieldDim.z), [&](int ind) {
			Point3D pt = this->ind2pt(ind);
			const Point3D ptC = const_cast<Point3D&>(pt);
			int i, j;

			if (!cellFieldG->get(ptC)) {

				std::vector<float> qtyOld = NCMaterialsField->get(ptC)->getNCMaterialsQuantityVecOld();
				std::vector<float> qtyNew = std::vector<float>(qtyOld);

				unsigned int numFields = fieldVec.size();
				std::vector<float> fieldValsOld = this->getFieldVals(ptC);
				std::vector<float> fieldValsNew = std::vector<float>(fieldValsOld);

				// General interactions
				if (GeneralInteractionsDefined) {

					NCMaterialsExpressionEvaluator *ee = this->getNCMExprEvalLocal((unsigned int)ind);
					std::tuple<std::vector<float>, std::vector<float> > incGenTuple = ee->evalExpressions(double(currentStep), qtyOld, fieldValsOld);

					if (qtyNew.size() > 0) {
						std::vector<float> qtyIncGen = std::get<0>(incGenTuple);
						for (unsigned int qtyIdx = 0; qtyIdx < qtyNew.size(); ++qtyIdx) qtyNew[qtyIdx] += qtyIncGen[qtyIdx];
					}

					if (numFields > 0) {
						std::vector<float> fieldIncGen = std::get<1>(incGenTuple);
						for (unsigned int fldIdx = 0; fldIdx < numFields; ++fldIdx) fieldValsNew[fldIdx] += fieldIncGen[fldIdx];
					}

				}

				// Material interactions

				if (MaterialInteractionsDefined) {

					// Material reactions with at site
					if (MaterialReactionsDefined)
						for each (i in idxMaterialReactionsDefined)
							for each (j in idxidxMaterialReactionsDefined[i])
								qtyNew[i] += materialReactionCoefficientsByIndex[i][j][0] * qtyOld[j];

					std::vector<float> qtyNeighbor;
					Point3D nPt;
					Neighbor neighbor;
					for (unsigned int nIdx = 0; nIdx <= nNeighbors; ++nIdx) {
						neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
						if (!neighbor.distance) continue;

						nPt = neighbor.pt;

						const Point3D nPtC = const_cast<Point3D&>(nPt);
						if (cellFieldG->get(nPtC)) continue; // Detect extracellular sites

						qtyNeighbor = NCMaterialsField->get(nPtC)->getNCMaterialsQuantityVecOld();

						// Material reactions with neighbors
						if (MaterialReactionsDefined)
							for each (i in idxMaterialReactionsDefined)
								for each (j in idxidxMaterialReactionsDefined[i])
									qtyNew[i] += materialReactionCoefficientsByIndex[i][j][1] * qtyNeighbor[j];

						// Material diffusion
						if (MaterialDiffusionDefined) {
							float materialDiffusionCoefficient;
							for each (i in idxMaterialDiffusionDefined) {
								materialDiffusionCoefficient = NCMaterialsVec->at(i).getMaterialDiffusionCoefficient();
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
						for each (j in idxidxFromFieldInteractionsDefined[i])
							qtyNew[i] += fromFieldReactionCoefficientsByIndex[i][j] * fieldValsOld[j];

					// Generate field source terms
					for each (i in idxToFieldInteractionsDefined)
						for each (j in idxidxToFieldInteractionsDefined[i])
							fieldValsNew[i] += toFieldReactionCoefficientsByIndex[i][j] * qtyOld[j];

				}

				NCMaterialsField->get(ptC)->setNCMaterialsQuantityVec(checkQuantities(qtyNew));
				for (unsigned int fieldIdx = 0; fieldIdx < fieldVec.size(); ++fieldIdx)
					fieldVec[fieldIdx]->set(ptC, max(0.0, fieldValsNew[fieldIdx]));

			}
		});
	}

	// Cell interactions
	if (CellInteractionsDefined) calculateCellInteractions(NCMaterialsField);

}

void NCMaterialsSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	// Do this to enable initializations independently of startup routine
	if (NCMaterialsInitialized) return;

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton);

	// Get NCMaterials plugin
	bool pluginAlreadyRegisteredFlag;
	ncMaterialsPlugin = (NCMaterialsPlugin*)Simulator::pluginManager.get("NCMaterials", &pluginAlreadyRegisteredFlag);
	if (!pluginAlreadyRegisteredFlag) return;

	// Forward steppable to plugin
	ncMaterialsPlugin->setNCMaterialsSteppable(this);

	// Get cell types
	cellTypeNames.clear();
	std::string cellTypeName;
	for (int j = 0; j <= (int)automaton->getMaxTypeId(); ++j) {
		
		cellTypeName = automaton->getTypeName(j);

		cerr << "   Got type " << cellTypeName << ": " << j << endl;

		cellTypeNames.insert(make_pair(automaton->getTypeName(j), j));
	}
	numberOfCellTypes = cellTypeNames.size();

	// Initialize preliminaries from NCMaterials plugin
	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();
	NCMaterialsField = ncMaterialsPlugin->getNCMaterialField();
	NCMaterialsVec = ncMaterialsPlugin->getNCMaterialsVecPtr();

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

	int materialIndex;
	int catalystIndex;
	
	MaterialInteractionsDefined = false;
	if (xmlData->findElement("MaterialInteraction")) {
		MaterialReactionsDefined = true;

		cerr << "Getting NCMaterial interactions..." << endl;

		for (int XMLidx = 0; XMLidx < MaterialInteractionXMLVec.size(); ++XMLidx) {

			thisMaterialName = MaterialInteractionXMLVec[XMLidx]->getAttribute("NCMaterial");

			cerr << "   NCMaterial " << thisMaterialName << endl;

			catalystName = MaterialInteractionXMLVec[XMLidx]->getFirstElement("Catalyst")->getText();
			cerr << "   Catalyst " << catalystName << endl;

			coeff = (float) MaterialInteractionXMLVec[XMLidx]->getFirstElement("ConstantCoefficient")->getDouble();
			cerr << "   Constant coefficient " << coeff << endl;

			int interactionOrder = 0;
			if (MaterialInteractionXMLVec[XMLidx]->findElement("NeighborhoodOrder")) {
				interactionOrder = MaterialInteractionXMLVec[XMLidx]->getFirstElement("NeighborhoodOrder")->getInt();
			}
			
			materialIndex = getNCMaterialIndexByName(thisMaterialName);

			ASSERT_OR_THROW("NCMaterial not defined in NCMaterials plugin.", materialIndex >= 0);

			catalystIndex = getNCMaterialIndexByName(catalystName);

			ASSERT_OR_THROW("Catalyst not defined in NCMaterials plugin.", catalystIndex >= 0);

			cerr << "   ";
			NCMaterialsVec->at(materialIndex).setMaterialReactionCoefficientByName(catalystName, coeff, interactionOrder);
			hasMaterialReactionsDefined[materialIndex] = true;

			idxidxMaterialReactionsDefined[materialIndex].push_back(catalystIndex);

		}

		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) if (hasMaterialReactionsDefined[mtlIdx]) idxMaterialReactionsDefined.push_back(mtlIdx); 

	}
	else MaterialReactionsDefined = false;

	if (xmlData->findElement("MaterialDiffusion")) {
		MaterialDiffusionDefined = true;

		cerr << "Getting NCMaterial diffusion..." << endl;

		for (int XMLidx = 0; XMLidx < MaterialDiffusionXMLVec.size(); ++XMLidx) {

			thisMaterialName = MaterialDiffusionXMLVec[XMLidx]->getAttribute("NCMaterial");

			cerr << "   NCMaterial " << thisMaterialName << endl;

			catalystName = thisMaterialName;

			coeff = (float)MaterialDiffusionXMLVec[XMLidx]->getDouble();
			cerr << "      Diffusion coefficient " << coeff << endl;

			ASSERT_OR_THROW("Invalid diffusion coefficient: must be positive.", coeff >= 0);

			materialIndex = getNCMaterialIndexByName(thisMaterialName);

			ASSERT_OR_THROW("NCMaterial not defined in NCMaterials plugin.", materialIndex >= 0);

			NCMaterialsVec->at(materialIndex).setMaterialDiffusionCoefficient(coeff);
			hasMaterialDiffusionDefined[materialIndex] = true;

		}
		
		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) if (hasMaterialDiffusionDefined[mtlIdx]) idxMaterialDiffusionDefined.push_back(mtlIdx);

	}
	else MaterialDiffusionDefined = false;

	MaterialInteractionsDefined = MaterialReactionsDefined || MaterialDiffusionDefined;

	if (MaterialInteractionsDefined) constructMaterialReactionCoefficients();

	// Get diffusion field info
	map<string, Field3D<float>*> & nameFieldMap = sim->getConcentrationFieldNameMap();

	fieldVec.clear();
	fieldNames.clear();
	for (std::map<string, Field3D<float>*>::iterator itr = nameFieldMap.begin(); itr != nameFieldMap.end(); ++itr) {
		fieldNames.insert(make_pair(itr->first, fieldNames.size()));
		fieldVec.push_back(itr->second);
	}
	numberOfFields = fieldVec.size();

	if (xmlData->findElement("FieldInteraction")) {
		FieldInteractionsDefined = true;

		cerr << "Getting NCMaterial field interactions..." << endl;

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
			materialIndex = getNCMaterialIndexByName(thisMaterialName);

			ASSERT_OR_THROW("NCMaterial " + thisMaterialName + " not registered in NCMaterials plugin.", materialIndex >= 0);

			// load valid info

			if (toField) {
				cerr << "      Mapping NCMaterial onto field" << endl;
				NCMaterialsVec->at(materialIndex).setToFieldReactionCoefficientByName(thisFieldName, coeff);
				hasToFieldInteractionsDefined[fieldIndex] = true;
				idxidxToFieldInteractionsDefined[fieldIndex].push_back(materialIndex);
			}
			else {
				cerr << "      Mapping field onto NCMaterial" << endl;
				NCMaterialsVec->at(materialIndex).setFromFieldReactionCoefficientByName(thisFieldName, coeff);
				hasFromFieldInteractionsDefined[materialIndex] = true;
				idxidxFromFieldInteractionsDefined[materialIndex].push_back(fieldIndex);
			}

		}

		for (int fldIdx = 0; fldIdx < numberOfFields; ++fldIdx)
			if (hasToFieldInteractionsDefined[fldIdx]) idxToFieldInteractionsDefined.push_back(fldIdx);
		
		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx)
			if (hasFromFieldInteractionsDefined[mtlIdx]) idxFromFieldInteractionsDefined.push_back(mtlIdx);

		constructFieldReactionCoefficients();

	}

	// Get general interaction expression evaluator info

	// Default symbols are names
	materialSyms = std::vector<std::string>(ncMaterialsPlugin->getMaterialNameVector());
	fieldSyms = std::vector<std::string>(numberOfFields);
	std::vector<std::string> fieldNamesVec(numberOfFields);
	for (std::map<std::string, int>::iterator mItr = fieldNames.begin(); mItr != fieldNames.end(); ++mItr) {
		fieldSyms[mItr->second] = mItr->first;
		fieldNamesVec[mItr->second] = mItr->first;
	}

	if (xmlData->findElement("GeneralFieldsInteractions")) {

		GeneralInteractionsDefined = true;

		cerr << "Getting NCMaterials general field interactions..." << endl;

		CC3DXMLElement *xmlInts = xmlData->getFirstElement("GeneralFieldsInteractions");
		for each (CC3DXMLElement *xmlSym in xmlInts->getElements("Symbol")) {
			std::string sym = xmlSym->getText();
			std::string name;
			if (xmlSym->findAttribute("Material")) {
				name = xmlSym->getAttribute("Material");
				int idx = getNCMaterialIndexByName(name);
				if (idx >= 0) materialSyms[idx] = sym;
				else cerr << "   Material name not recognized for general field interaction: " << name << endl;
			}
			else if (xmlSym->findAttribute("Field")) {
				name = xmlSym->getAttribute("Field");
				int idx = getFieldIndexByName(name);
				if (idx >= 0) fieldSyms[idx] = sym;
				else cerr << "   Field name not recognized for general field interaction: " << name << endl;
			}
		}

		cerr << "   Processing expressions... " << endl;

		materialExprs = std::vector<std::string>(numberOfMaterials, "0.0");
		fieldExprs = std::vector<std::string>(numberOfFields, "0.0");
		std::vector<bool> materialExprsDefined = std::vector<bool>(numberOfMaterials, false);
		std::vector<bool> fieldExprsDefined = std::vector<bool>(numberOfFields, false);

		for each (CC3DXMLElement *xmlInt in xmlInts->getElements("Interaction")) {

			std::string expr = xmlInt->getText();
			std::string productName;
			int idx;

			if (xmlInt->findAttribute("Material")) {
				productName = xmlInt->getAttribute("Material");
				idx = getNCMaterialIndexByName(productName);
				if (idx >= 0) {
					if (materialExprsDefined[idx]) materialExprs[idx] += "+" + expr;
					else {
						materialExprsDefined[idx] = true;
						materialExprs[idx] = expr;
					}
				}
				else cerr << "   Material name not recognized for general field interaction: " << productName << endl;
			}
			else if (xmlInt->findAttribute("Field")) {
				productName = xmlInt->getAttribute("Field");
				std::map<std::string, int>::iterator mItr = fieldNames.find(productName);
				if (mItr != fieldNames.end()) {
					idx = mItr->second;
					if (fieldExprsDefined[idx]) fieldExprs[idx] += "+" + expr;
					else {
						fieldExprsDefined[idx] = true;
						fieldExprs[idx] = expr;
					}

				}
				else cerr << "   Field name not recognized for general field interaction: " << productName << endl;
			}

		}

		cerr << "   Got material expressions: " << endl;

		for (unsigned int symIdx = 0; symIdx < materialExprs.size(); ++symIdx) {

			cerr << "      " << getNCMaterialNameByIndex(symIdx) << "->" << materialExprs[symIdx] << endl;

		}

		cerr << "   Got field expressions: " << endl;

		for (unsigned int symIdx = 0; symIdx < fieldExprs.size(); ++symIdx) {

			cerr << "      " << fieldNamesVec[symIdx] << "->" << fieldExprs[symIdx] << endl;

		}

		cerr << "   Generating field expressions evaluators... ";

		NCMExprEvalLocal = std::vector<NCMaterialsExpressionEvaluator *>(fieldDim.x*fieldDim.y*fieldDim.z, nullptr);
		for (unsigned int ind = 0; ind < fieldDim.x*fieldDim.y*fieldDim.z; ++ind) {
			Point3D pt = ind2pt(ind);

			NCMaterialsExpressionEvaluator *_ee = new NCMaterialsExpressionEvaluator(numberOfMaterials, numberOfFields);
			_ee->init(pt);
			_ee->setSymbols(materialSyms, fieldSyms);

			for (unsigned int i = 0; i < numberOfMaterials; ++i)
				_ee->setMaterialExpression(i, materialExprs[i]);

			for (unsigned int i = 0; i < numberOfFields; ++i)
				_ee->setFieldExpression(i, fieldExprs[i]);

			NCMExprEvalLocal[ind] = _ee;

		}

		cerr << "done." << endl;

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

			ASSERT_OR_THROW("An NCMaterial must be defined for each cell interaction (using NCMaterial).", CellInteractionXMLVec[XMLidx]->findElement("NCMaterial"));
			thisMaterialName = CellInteractionXMLVec[XMLidx]->getFirstElement("NCMaterial")->getText();
			cerr << "      NCMaterial " << thisMaterialName << endl;

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
			materialIndex = getNCMaterialIndexByName(thisMaterialName);
			ASSERT_OR_THROW("NCMaterial " + thisMaterialName + " not registered.", materialIndex >= 0);
			ASSERT_OR_THROW("Probability must be non-negative", coeff >= 0);
			cellTypeIndex = getCellTypeIndexByName(thisCellType);
			ASSERT_OR_THROW("Cell type " + thisCellType + " not registered.", cellTypeIndex >= 0);
			cellTypeNewIndex = getCellTypeIndexByName(thisCellTypeNew);
			if (methodName == "Differentiation")
				ASSERT_OR_THROW("New cell type " + thisCellTypeNew + " not registered.", cellTypeNewIndex >= 0);

			if (methodName == "Proliferation")
				NCMaterialsVec->at(materialIndex).setCellTypeCoefficientsProliferation(thisCellType, coeff, thisCellTypeNew);
			
			else if (methodName == "Differentiation")
				NCMaterialsVec->at(materialIndex).setCellTypeCoefficientsDifferentiation(thisCellType, coeff, thisCellTypeNew);
			
			else if (methodName == "Death")
				NCMaterialsVec->at(materialIndex).setCellTypeCoefficientsDeath(thisCellType, coeff);
			
			else ASSERT_OR_THROW("Undefined cell response method: use Proliferation, Differentiation, or Death.", false);

			hasCellInteractionsDefined[materialIndex] = true;

		}

		for (int mtlIdx = 0; mtlIdx < numberOfMaterials; ++mtlIdx) if (hasCellInteractionsDefined[mtlIdx]) idxCellInteractionsDefined.push_back(mtlIdx);

		constructCellTypeCoefficients();
	}

	NCMaterialsInitialized = true;
	AnyInteractionsDefined = MaterialInteractionsDefined || FieldInteractionsDefined || GeneralInteractionsDefined || CellInteractionsDefined;
}

void NCMaterialsSteppable::calculateCellInteractions(Field3D<NCMaterialsData *> *NCMaterialsField) {

	ASSERT_OR_THROW("NCMaterials steppable not yet initialized.", NCMaterialsInitialized);

	ncMaterialsPlugin->resetCellResponse();
	
	CellG * cell = 0;

	CellInventory::cellInventoryIterator cInvItr;
	CellInventory * cellInventoryPtr = &potts->getCellInventory();
	int nNeighbors = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

	// Initialize information for counting cell interactions

	int numberOfCells = cellInventoryPtr->getSize();
	concurrency::combinable<std::vector<std::tuple<CellG*, int, std::string> > > cellResponses([] {
		return std::vector<std::tuple<CellG*, int, std::string> >(); });
	concurrency::concurrent_vector<CellG*> cellVec = concurrency::concurrent_vector<CellG*>(numberOfCells);

	int idx = 0;
	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
		cellVec[idx] = cInvItr->second;
		++idx;
	}

	//		randomly order evaluation of responses
	std::vector<std::vector<int> > responseOrderSet = permutations(4);
	std::vector<int> responseOrder = responseOrderSet[rand() % responseOrderSet.size()];

	// Loop over cells and consider each response
	std::vector<std::string> actionOrderedVec = std::vector<std::string>{ "Proliferation", "Death", "Differentiation", "Proliferation" };
	concurrency::parallel_for_each(cellVec.begin(), cellVec.end(), [&](CellG* cell) {
		std::tuple<float, float, std::vector<float>, std::vector<float> > probs = calculateCellProbabilities(cell, NCMaterialsField);

		float probAccumProlif = std::get<0>(probs);
		float probAccumDeath = std::get<1>(probs);
		std::vector<float> probAccumDiff = std::get<2>(probs);
		std::vector<float> probAccumProlifAsym = std::get<3>(probs);

		// Consider each response in random order of response
		bool resp = false;
		std::string CellTypeDiff = "";
		std::map<std::string, int>::iterator mitr;
		for each (int respIdx in responseOrder) {
			switch (respIdx) {
			case 0: // proliferation
				resp = testProb(probAccumProlif);
				break;
			case 1: // death
				resp = testProb(probAccumDeath);
				break;
			case 2: // differentiation
				for (mitr = cellTypeNames.begin(); mitr != cellTypeNames.end(); ++mitr) {
					resp = testProb(probAccumDiff[mitr->second]);
					CellTypeDiff = mitr->first;
					if (resp) break;
				}
				break;
			case 3: // asymmetric division
				for (mitr = cellTypeNames.begin(); mitr != cellTypeNames.end(); ++mitr) {
					resp = testProb(probAccumProlifAsym[mitr->second]);
					CellTypeDiff = mitr->first;
					if (resp) break;
				}
				break;
			}
			if (resp) {
				cellResponses.local().push_back(make_tuple(cell, respIdx, CellTypeDiff));
				break;
			}
		}

	});

	std::vector<std::tuple<CellG*, int, std::string> > combinedResponsesAccepted;
	cellResponses.combine_each([&](std::vector<std::tuple<CellG*, int, std::string> > local) {
		for each(std::tuple<CellG*, int, std::string> cellResponse in local)
			combinedResponsesAccepted.push_back(cellResponse); });

	for each(std::tuple<CellG*, int, std::string> cellResponse in combinedResponsesAccepted)
		ncMaterialsPlugin->addCellResponse(std::get<0>(cellResponse), actionOrderedVec[std::get<1>(cellResponse)], std::get<2>(cellResponse));

}

std::tuple<float, float, std::vector<float>, std::vector<float> > NCMaterialsSteppable::calculateCellProbabilities(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField, std::vector<bool> _resultsSelect) {
	if (!_ncmaterialsField) _ncmaterialsField = NCMaterialsField;

	float probAccumProlif = 0.0;
	std::vector<float> probAccumProlifAsym(numberOfCellTypes, 0.0);
	std::vector<float> probAccumDiff(numberOfCellTypes, 0.0);
	float probAccumDeath = 0.0;

	if (!cell) return make_tuple(probAccumProlif, probAccumDeath, probAccumDiff, probAccumProlifAsym);

	int nNeighbors = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
	Neighbor neighbor;
	CellG *nCell = 0;
	Point3D pt;
	int cellType = (int)cell->type;
	std::vector<float> qtyOld(numberOfMaterials, 0.0);

	// Loop over cell boundaries to find interfaces with NCMaterials
	std::set<BoundaryPixelTrackerData > *boundarySet = boundaryTrackerPlugin->getPixelSetForNeighborOrderPtr(cell, neighborOrder);
	for (std::set<BoundaryPixelTrackerData >::iterator bInvItr = boundarySet->begin(); bInvItr != boundarySet->end(); ++bInvItr) {

		// Loop over cell boundary neighbors and accumulate probabilities from medium sites
		pt = bInvItr->pixel;

		for (unsigned int nIdx = 0; nIdx <= nNeighbors; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

			if (!neighbor.distance) continue;

			nCell = potts->getCellFieldG()->get(neighbor.pt);

			if (nCell) continue;

			qtyOld = _ncmaterialsField->get(neighbor.pt)->getNCMaterialsQuantityVecOld();

			for each (int i in idxCellInteractionsDefined) {
				if (_resultsSelect[0]) probAccumProlif += CellTypeCoefficientsProliferationByIndex[i][cellType] * qtyOld[i];
				if (_resultsSelect[1]) probAccumDeath += CellTypeCoefficientsDeathByIndex[i][cellType] * qtyOld[i];
				if (_resultsSelect[2] && _resultsSelect[3]) {
					for (int typeIdx = 0; typeIdx < probAccumDiff.size(); ++typeIdx) {
						probAccumProlifAsym[typeIdx] += CellTypeCoefficientsProliferationAsymByIndex[i][cellType][typeIdx] * qtyOld[i];
						probAccumDiff[typeIdx] += CellTypeCoefficientsDifferentiationByIndex[i][cellType][typeIdx] * qtyOld[i];
					}
				}
				else if (_resultsSelect[2])
					for (int typeIdx = 0; typeIdx < probAccumDiff.size(); ++typeIdx) 
						probAccumDiff[typeIdx] += CellTypeCoefficientsDifferentiationByIndex[i][cellType][typeIdx] * qtyOld[i];
				else if (_resultsSelect[3]) 
					for (int typeIdx = 0; typeIdx < probAccumDiff.size(); ++typeIdx) 
						probAccumProlifAsym[typeIdx] += CellTypeCoefficientsProliferationAsymByIndex[i][cellType][typeIdx] * qtyOld[i];
			}

		}

	}

	return make_tuple(probAccumProlif, probAccumDeath, probAccumDiff, probAccumProlifAsym);
}

float NCMaterialsSteppable::calculateCellProbabilityProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return std::get<0>(calculateCellProbabilities(cell, _ncmaterialsField, { true, false, false, false }));
}

float NCMaterialsSteppable::calculateCellProbabilityDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return std::get<1>(calculateCellProbabilities(cell, _ncmaterialsField, { false, true, false, false }));
}

float NCMaterialsSteppable::calculateCellProbabilityDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	int newCellTypeIndex = getCellTypeIndexByName(newCellType);
	if (newCellTypeIndex < 0) return 0.0;
	return std::get<2>(calculateCellProbabilities(cell, _ncmaterialsField, { false, false, true, false }))[newCellTypeIndex];
}

float NCMaterialsSteppable::calculateCellProbabilityAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	int newCellTypeIndex = getCellTypeIndexByName(newCellType);
	if (newCellTypeIndex < 0) return 0.0;
	return std::get<3>(calculateCellProbabilities(cell, _ncmaterialsField, { false, false, false, true }))[newCellTypeIndex];
}

bool NCMaterialsSteppable::getCellResponseProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return testProb(calculateCellProbabilityProliferation(cell, _ncmaterialsField));
}

bool NCMaterialsSteppable::getCellResponseDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return testProb(calculateCellProbabilityDeath(cell, _ncmaterialsField));
}

bool NCMaterialsSteppable::getCellResponseDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return testProb(calculateCellProbabilityDifferentiation(cell, newCellType, _ncmaterialsField));
}

bool NCMaterialsSteppable::getCellResponseAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return testProb(calculateCellProbabilityAsymmetricDivision(cell, newCellType, _ncmaterialsField));
}

float NCMaterialsSteppable::calculateTotalInterfaceQuantityByMaterialIndex(CellG *cell, int _materialIdx, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	if (!_ncmaterialsField) _ncmaterialsField = NCMaterialsField;

	float qty_total = 0.0;
	
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	int nNeighbors = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
	Neighbor neighbor;
	CellG *nCell = 0;
	Point3D pt;
	int cellId = (int)cell->id;
	int cellType = (int)cell->type;
	std::vector<float> qtyOld(numberOfMaterials, 0.0);

	// Loop over cell boundaries to find interfaces with NCMaterials
	std::set<BoundaryPixelTrackerData > *boundarySet = boundaryTrackerPlugin->getPixelSetForNeighborOrderPtr(cell, neighborOrder);
	for (std::set<BoundaryPixelTrackerData >::iterator bInvItr = boundarySet->begin(); bInvItr != boundarySet->end(); ++bInvItr) {

		// Loop over cell boundary neighbors and accumulate probabilities from medium sites
		pt = bInvItr->pixel;

		for (unsigned int nIdx = 0; nIdx <= nNeighbors; ++nIdx) {
			neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

			if (!neighbor.distance) continue;

			nCell = cellFieldG->get(neighbor.pt);

			if (nCell) continue;

			qtyOld = _ncmaterialsField->get(neighbor.pt)->getNCMaterialsQuantityVec();

			qty_total += qtyOld[_materialIdx];

		}

	}

	return qty_total;
}

float NCMaterialsSteppable::calculateTotalInterfaceQuantityByMaterialName(CellG *cell, std::string _materialName, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return calculateTotalInterfaceQuantityByMaterialIndex(cell, getNCMaterialIndexByName(_materialName), _ncmaterialsField);
}

void NCMaterialsSteppable::constructFieldReactionCoefficients() {

	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();
	numberOfFields = fieldVec.size();

	toFieldReactionCoefficientsByIndex = std::vector<std::vector<float> >(numberOfFields, std::vector<float>(numberOfMaterials, 0.0));
	fromFieldReactionCoefficientsByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfFields, 0.0));

	for (std::vector<int>::iterator i = idxFromFieldInteractionsDefined.begin(); i != idxFromFieldInteractionsDefined.end(); ++i) {
		for (std::vector<int>::iterator j = idxidxFromFieldInteractionsDefined[*i].begin(); j != idxidxFromFieldInteractionsDefined[*i].end(); ++j) {
			fromFieldReactionCoefficientsByIndex[*i][*j] = NCMaterialsVec->at(*i).getFromFieldReactionCoefficientByName(getFieldNameByIndex(*j));
		}
	}
	for (std::vector<int>::iterator i = idxToFieldInteractionsDefined.begin(); i != idxToFieldInteractionsDefined.end(); ++i) {
		for (std::vector<int>::iterator j = idxidxToFieldInteractionsDefined[*i].begin(); j != idxidxToFieldInteractionsDefined[*i].end(); ++j) {
			toFieldReactionCoefficientsByIndex[*i][*j] = NCMaterialsVec->at(*j).getToFieldReactionCoefficientByName(getFieldNameByIndex(*i));
		}
	}

}

void NCMaterialsSteppable::constructMaterialReactionCoefficients() {

	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();

	materialReactionCoefficientsByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(2, 0.0)));

	for (std::vector<int>::iterator i = idxMaterialReactionsDefined.begin(); i != idxMaterialReactionsDefined.end(); ++i) {
		for (std::vector<int>::iterator j = idxidxMaterialReactionsDefined[*i].begin(); j != idxidxMaterialReactionsDefined[*i].end(); ++j) {
			materialReactionCoefficientsByIndex[*i][*j] = NCMaterialsVec->at(*i).getMaterialReactionCoefficientsByName(NCMaterialsVec->at(*j).getName());
		}
	}
}

void NCMaterialsSteppable::constructCellTypeCoefficients() {

	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsProliferationByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));
	CellTypeCoefficientsProliferationAsymByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfCellTypes, std::vector<float>(numberOfCellTypes, 0.0)));
	CellTypeCoefficientsDifferentiationByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfCellTypes, std::vector<float>(numberOfCellTypes, 0.0)));
	CellTypeCoefficientsDeathByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));

	for each (int i in idxCellInteractionsDefined) {
		for (int j = 0; j < CellTypeCoefficientsProliferationByIndex[i].size(); ++j) {

			std::string cellTypeName = automaton->getTypeName(j);

			CellTypeCoefficientsProliferationByIndex[i][j] = NCMaterialsVec->at(i).getCellTypeCoefficientsProliferation(cellTypeName);
			CellTypeCoefficientsDeathByIndex[i][j] = NCMaterialsVec->at(i).getCellTypeCoefficientsDeath(cellTypeName);
			for (int k = 0; k < CellTypeCoefficientsDifferentiationByIndex[i][j].size(); ++k) {
				CellTypeCoefficientsProliferationAsymByIndex[i][j][k] = NCMaterialsVec->at(i).getCellTypeCoefficientsProliferationAsymmetric(cellTypeName, automaton->getTypeName(k));
				CellTypeCoefficientsDifferentiationByIndex[i][j][k] = NCMaterialsVec->at(i).getCellTypeCoefficientsDifferentiation(cellTypeName, automaton->getTypeName(k));
			}
		}
	}

}

void NCMaterialsSteppable::constructCellTypeCoefficientsProliferation() {

	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsProliferationByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));

	for (std::vector<int>::iterator i = idxCellInteractionsDefined.begin(); i != idxCellInteractionsDefined.end(); ++i) {
		for (int j = 0; j < numberOfCellTypes; ++j) {
			CellTypeCoefficientsProliferationByIndex[*i][j] = NCMaterialsVec->at(*i).getCellTypeCoefficientsProliferation(automaton->getTypeName(j));
		}
	}

}

void NCMaterialsSteppable::constructCellTypeCoefficientsDifferentiation() {

	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsDifferentiationByIndex = std::vector<std::vector<std::vector<float> > >(numberOfMaterials, std::vector<std::vector<float> >(numberOfCellTypes, std::vector<float>(numberOfCellTypes, 0.0)));

	for (std::vector<int>::iterator i = idxCellInteractionsDefined.begin(); i != idxCellInteractionsDefined.end(); ++i) {
		for (int j = 0; j < numberOfCellTypes; ++j) {
			for (int k = 0; k < numberOfCellTypes; ++k) {
				CellTypeCoefficientsDifferentiationByIndex[*i][j][k] = NCMaterialsVec->at(*i).getCellTypeCoefficientsDifferentiation(automaton->getTypeName(j), automaton->getTypeName(k));
			}
		}
	}

}

void NCMaterialsSteppable::constructCellTypeCoefficientsDeath() {

	numberOfMaterials = ncMaterialsPlugin->getNumberOfMaterials();

	CellTypeCoefficientsDeathByIndex = std::vector<std::vector<float> >(numberOfMaterials, std::vector<float>(numberOfCellTypes, 0.0));

	for (std::vector<int>::iterator i = idxCellInteractionsDefined.begin(); i != idxCellInteractionsDefined.end(); ++i) {
		for (int j = 0; j < numberOfCellTypes; ++j) {
			CellTypeCoefficientsDeathByIndex[*i][j] = NCMaterialsVec->at(*i).getCellTypeCoefficientsDeath(automaton->getTypeName(j));
		}
	}

}

std::vector<float> NCMaterialsSteppable::getFieldVals(const Point3D &_pt) {
	if (numberOfFields == 0) return std::vector<float>();

	std::vector<float> _fieldVals = std::vector<float>(numberOfFields, 0.0);
	for (unsigned int fieldIdx = 0; fieldIdx < numberOfFields; ++fieldIdx) _fieldVals[fieldIdx] = fieldVec[fieldIdx]->get(_pt);
	return _fieldVals;
}

void NCMaterialsSteppable::calculateMaterialToFieldInteractions(const Point3D &pt, std::vector<float> _qtyOld) {
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

std::vector<std::vector<int> > NCMaterialsSteppable::permutations(int _numberOfVals) {
	std::vector<std::vector<int> > permutationVec;
	std::vector<int> permutationSet(_numberOfVals, 0);
	for (int i = 0; i < _numberOfVals; ++i) permutationSet[i] = i;
	do {
		permutationVec.push_back(permutationSet);
	} while (next_permutation(permutationSet.begin(), permutationSet.end()));
	return permutationVec;
}

Point3D NCMaterialsSteppable::ind2pt(unsigned int _ind) {
	Point3D pt;
	pt.x = _ind % fieldDim.x;
	pt.y = (_ind - pt.x) / fieldDim.x % fieldDim.y;
	pt.z = ((_ind - pt.x) / fieldDim.x - pt.y) / fieldDim.y;
	return pt;
}

unsigned int NCMaterialsSteppable::pt2ind(const Point3D &_pt, Dim3D _fieldDim) {
	int ind = _pt.x + _fieldDim.x * (_pt.y + _fieldDim.y * _pt.z);
	if (ind < 0 || ind > fieldDim.x*fieldDim.y*fieldDim.z) { throw BasicException("Point is not valid."); }
	return (unsigned int)(ind);
}

std::string NCMaterialsSteppable::toString(){
   return "NCMaterialsSteppable";
}

std::string NCMaterialsSteppable::steerableName(){
   return toString();
}