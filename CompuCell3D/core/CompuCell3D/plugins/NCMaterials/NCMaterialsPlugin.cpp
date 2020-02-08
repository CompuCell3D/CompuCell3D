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

/**
@author T.J. Sego, Ph.D.
*/

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include "NCMaterialsPlugin.h"
#include "PublicUtilities/Vector3.h"
#include <CompuCell3D/steppables/NCMaterialsSteppable/NCMaterialsSteppable.h>


NCMaterialsPlugin::NCMaterialsPlugin() :
    pUtils(0),
    lockPtr(0),
    xmlData(0),
    numberOfMaterials(0),
    weightDistance(false),
    NCMaterialsInitialized(false)
{}

NCMaterialsPlugin::~NCMaterialsPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;

	deleteNCMaterialsField();
}

void NCMaterialsPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
	automaton = potts->getAutomaton();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    cerr << "Registering NCMaterials cell attributes..." << endl;

    potts->getCellFactoryGroupPtr()->registerClass(&NCMaterialCellDataAccessor);

	cerr << "Registering NCMaterials plugin..." << endl;

    potts->registerEnergyFunctionWithName(this, "NCMaterials");
    potts->registerCellGChangeWatcher(this);
    simulator->registerSteerableObject(this);

    if (xmlData->getFirstElement("WeightEnergyByDistance")) {
        weightDistance = true;
    }

	// Boundary strategy for adhesion; based on implementation in AdhesionFlex
	boundaryStrategyAdh = BoundaryStrategy::getInstance();
	maxNeighborIndexAdh = 0;

	if (xmlData->getFirstElement("Depth")) {
		maxNeighborIndexAdh = boundaryStrategyAdh->getMaxNeighborIndexFromDepth(xmlData->getFirstElement("Depth")->getDouble());
	}
	else {
		if (xmlData->getFirstElement("NeighborOrder")) {
			maxNeighborIndexAdh = boundaryStrategyAdh->getMaxNeighborIndexFromNeighborOrder(xmlData->getFirstElement("NeighborOrder")->getUInt());
		}
		else {
			maxNeighborIndexAdh = boundaryStrategyAdh->getMaxNeighborIndexFromNeighborOrder(1);
		}
	}

    // Gather XML user specifications
    CC3DXMLElementList NCMaterialNameXMLVec = _xmlData->getElements("NCMaterial");
    NCMaterialAdhesionXMLVec = _xmlData->getElements("NCAdhesion");
    CC3DXMLElementList NCMaterialAdvectionBoolXMLVec = _xmlData->getElements("NCMaterialAdvects");
    CC3DXMLElementList NCMaterialDurabilityXMLVec = _xmlData->getElements("NCMaterialDurability");
	CC3DXMLElementList NCMaterialDiffusivityXMLVec = _xmlData->getElements("NCMaterialDiffusivity");
    NCMaterialRemodelingQuantityXMLVec = _xmlData->getElements("RemodelingQuantity");

    // generate name->integer index map for NC materials
    // generate array of pointers to NC materials according to user specification
    // assign material name from user specification
    set<std::string> NCMaterialNameSet;
    NCMaterialsVec.clear();
    NCMaterialNameIndexMap.clear();

    std::string NCMaterialName;

	cerr << "Declaring NCMaterials... " << endl;

    for (int i = 0; i < NCMaterialNameXMLVec.size(); ++i) {
        NCMaterialName = NCMaterialNameXMLVec[i]->getAttribute("Material");

		cerr << "   NCMaterial " << i << ": " << NCMaterialName << endl;

        if (!NCMaterialNameSet.insert(NCMaterialName).second) {
            ASSERT_OR_THROW(string("Duplicate NCMaterial Name=") + NCMaterialName + " specified in NCMaterials section ", false);
            continue;
        }

        NCMaterialsVec.push_back(NCMaterialComponentData());
        NCMaterialsVec[i].setName(NCMaterialName);

        NCMaterialNameIndexMap.insert(make_pair(NCMaterialName, i));
    }

    numberOfMaterials = NCMaterialsVec.size();

	cerr << "Number of NCMaterials defined: " << numberOfMaterials << endl;

    int NCMaterialIdx;
	bool NCMaterialIsAdvecting;

    // Assign optional advection specifications

	cerr << "Checking material advection options..." << endl;

    if (NCMaterialAdvectionBoolXMLVec.size() > 0){
        for (int i = 0; i < NCMaterialAdvectionBoolXMLVec.size(); i++) {
			NCMaterialName = NCMaterialAdvectionBoolXMLVec[i]->getAttribute("Material");
			NCMaterialIsAdvecting = NCMaterialAdvectionBoolXMLVec[i]->getBool();

			cerr << "   NCMaterial " << NCMaterialName << " advection mode: " << NCMaterialIsAdvecting << endl;

            NCMaterialIdx = getNCMaterialIndexByName(NCMaterialName);
            NCMaterialsVec[NCMaterialIdx].setTransferable(NCMaterialIsAdvecting);
        }
    }

    // Assign barrier Lagrange multiplier to each material from user specification

	float durabilityLM;

	cerr << "Assigning NCMaterial durability coefficients..." << endl;

    if (NCMaterialDurabilityXMLVec.size() > 0) {
        for (int i = 0; i < NCMaterialDurabilityXMLVec.size(); ++i) {
			NCMaterialName = NCMaterialDurabilityXMLVec[i]->getAttribute("Material");
			durabilityLM = (float)NCMaterialDurabilityXMLVec[i]->getDouble();

			cerr << "   NCMaterial " << NCMaterialName << " barrier Lagrange multiplier: " << durabilityLM << endl;
			
			NCMaterialIdx = getNCMaterialIndexByName(NCMaterialName);
            NCMaterialsVec[NCMaterialIdx].setDurabilityLM(durabilityLM);
        }
    }

	/* 
	Assign field diffusion coefficients to each material from user specification
		Fields with incomplete specification throw and error for feedback to user
		Fields with no specification have no flag for diffusion solvers
		Fields with complete specification have flag for diffusion solvers 
	*/
	if (NCMaterialDiffusivityXMLVec.size() > 0) {

		// Get user specs

		cerr << "Assigning NCMaterial diffusion coefficients..." << endl;

		for (int i = 0; i < NCMaterialDiffusivityXMLVec.size(); ++i) {
			NCMaterialName = NCMaterialDiffusivityXMLVec[i]->getAttribute("Material");
			NCMaterialIdx = getNCMaterialIndexByName(NCMaterialName);
			ASSERT_OR_THROW("NCMaterial " + NCMaterialName + " not defined.", NCMaterialIdx >= 0);

			std::string fieldName = NCMaterialDiffusivityXMLVec[i]->getAttribute("Field");
			float coeff = (float) NCMaterialDiffusivityXMLVec[i]->getDouble();

			ASSERT_OR_THROW("NCMaterial diffusion coefficients must be positive.", coeff >= 0);

			cerr << "   (" << NCMaterialName << ", " << fieldName << "): " << coeff << endl;

			NCMaterialsVec[NCMaterialIdx].setFieldDiffusivity(fieldName, coeff);

			setVariableDiffusivityFieldFlagMap(fieldName, true);

		}

		// Test user specs

		for (std::map<std::string, bool>::iterator mitr = variableDiffusivityFieldFlagMap.begin(); mitr != variableDiffusivityFieldFlagMap.end(); ++mitr) {
			std::string fieldName = mitr->first;
			std::vector<std::string> materialsNotDefined;

			cerr << "Checking completion for field " << fieldName << "..." << endl;

			for (int i = 0; i < numberOfMaterials; ++i) if (NCMaterialsVec[i].getFieldDiffusivity(fieldName) < 0.0) materialsNotDefined.push_back(NCMaterialsVec[i].getName());
			if (materialsNotDefined.size() > 0) {
				for (int i = 0; i < materialsNotDefined.size(); ++i) {
					cerr << "   Diffusion coefficient not found for NCMaterial " << materialsNotDefined[i] << endl;
				}
			}
			ASSERT_OR_THROW("If specifying diffusion coefficients for a field in NCMaterials, the diffusion coefficient for each NCMaterial must be specified.", materialsNotDefined.size() == 0)
		}

	}

    // Initialize quantity vector field

	cerr << "Initializing NCMaterials quantity field..." << endl;

    fieldDim=potts->getCellFieldG()->getDim();

	initializeNCMaterialsField(fieldDim);

	// Field design, if specified

	if (_xmlData->findElement("FieldDesign")) {

		cerr << "Initializing field designs" << endl;

		CC3DXMLElementList NCMaterialFieldDesignVec = _xmlData->getElements("FieldDesign");
		
		// in case user specified multiple FieldDesign elements, loop over all discovered

		bool designIsValid;

		for (int fieldDesignIndex = 0; fieldDesignIndex < NCMaterialFieldDesignVec.size(); ++fieldDesignIndex) {

			CC3DXMLElementList FieldDesigns = NCMaterialFieldDesignVec[fieldDesignIndex]->getElements("Design");

			// apply design

			for (CC3DXMLElementList::iterator designItr = FieldDesigns.begin(); designItr != FieldDesigns.end(); ++designItr) {

				// get specified materials

				CC3DXMLElementList theseMaterials = (*designItr)->getElements("NCMaterial");
				std::vector<float> thisQuantityVector(numberOfMaterials, 0.0);

				bool aMaterialWasSpecified = false;

				for (CC3DXMLElementList::iterator materialItr = theseMaterials.begin(); materialItr != theseMaterials.end(); ++materialItr) {
					int materialIndex = getNCMaterialIndexByName((*materialItr)->getAttribute("Material"));
					if (materialIndex < 0) { continue; }
					aMaterialWasSpecified = true;
					thisQuantityVector[materialIndex] = (float)(*materialItr)->getDouble();
				}

				if (!aMaterialWasSpecified) { continue; }

				std::vector<float> thisQuantityVectorChecked = checkQuantities(thisQuantityVector);

				cerr << "   Drawing with quantity vector (";
				for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
					cerr << thisQuantityVectorChecked[materialIndex];
					if (materialIndex < numberOfMaterials - 1) { cerr << ", "; }
					else { cerr << ")" << endl; }
				}
				
				// apply method

				ASSERT_OR_THROW("A method must be specified for a design.", (*designItr)->findAttribute("Method"));
				std::string thisMethod = (*designItr)->getAttribute("Method");

				if (thisMethod == "Ellipse") {

					cerr << "   Drawing ellipse." << endl;

					int length = 1.0;
					Point3D center(0, 0, 0);
					double angle = 0.0;
					double eccentricity = 0.0;

					ASSERT_OR_THROW("One center point must be defined per ellipse (using CenterPoint).", (*designItr)->findElement("CenterPoint"));
					CC3DXMLElementList FocusElement = (*designItr)->getElements("CenterPoint");
					ASSERT_OR_THROW("Only one center point can be defined per ellipse.", FocusElement.size() == 1);
					if ((*FocusElement[0]).findAttribute("x")) { center.x = (*FocusElement[0]).getAttributeAsShort("x"); }
					if ((*FocusElement[0]).findAttribute("y")) { center.y = (*FocusElement[0]).getAttributeAsShort("y"); }

					ASSERT_OR_THROW("Ellipse semi-major axis half length must be defined (using MajorHalfLength)", (*designItr)->findElement("MajorHalfLength"));
					CC3DXMLElementList LengthElement = (*designItr)->getElements("MajorHalfLength");
					ASSERT_OR_THROW("Only one semi-major axis half length can be defined per ellipse.", LengthElement.size() == 1);
					length = (*LengthElement[0]).getInt();

					if ((*designItr)->findElement("Eccentricity")) {

						CC3DXMLElementList EccentricityElement = (*designItr)->getElements("Eccentricity");
						ASSERT_OR_THROW("Only one eccentricity can be defined per ellipse.", EccentricityElement.size() == 1);
						eccentricity = (*EccentricityElement[0]).getDouble();
						ASSERT_OR_THROW("Ellipse eccentricity must be between zero and one.", eccentricity > 0 && eccentricity < 1);

					}

					if ((*designItr)->findElement("RotationAngle")) {

						CC3DXMLElementList RotationAngleElement = (*designItr)->getElements("RotationAngle");
						ASSERT_OR_THROW("Only one rotation angle can be defined per ellipse.", RotationAngleElement.size() == 1);
						angle = (*RotationAngleElement[0]).getDouble();

					}

					EllipseDraw(thisQuantityVectorChecked, length, center, angle, eccentricity);

				}
				else if (thisMethod == "Circle") {

					cerr << "   Drawing circle." << endl;

					int length = 1.0;
					Point3D center(0, 0, 0);

					ASSERT_OR_THROW("One center point must be defined per circle (using CenterPoint).", (*designItr)->findElement("CenterPoint"));
					CC3DXMLElementList FocusElement = (*designItr)->getElements("CenterPoint");
					ASSERT_OR_THROW("Only one center point can be defined per circle.", FocusElement.size() == 1);
					if ((*FocusElement[0]).findAttribute("x")) { center.x = (*FocusElement[0]).getAttributeAsShort("x"); }
					if ((*FocusElement[0]).findAttribute("y")) { center.y = (*FocusElement[0]).getAttributeAsShort("y"); }

					ASSERT_OR_THROW("Circle radius must be defined (using Radius)", (*designItr)->findElement("Radius"));
					CC3DXMLElementList RadiusElement = (*designItr)->getElements("Radius");
					ASSERT_OR_THROW("Only radius can be defined per circle.", RadiusElement.size() == 1);
					length = (*RadiusElement[0]).getInt();

					EllipseDraw(thisQuantityVectorChecked, length, center, 0.0, 0.0);

				}
				else if (thisMethod == "Parallelepiped") {

					cerr << "   Drawing parallelepiped." << endl;
					
					Point3D startPos(0, 0, 0);
					std::vector<Point3D> lenVec(3, Point3D(0, 0, 0));
					lenVec[0].x = -1;
					lenVec[1].y = -1;
					lenVec[2].z = -1;

					ASSERT_OR_THROW("One start point must be defined per parallelepiped (using StartPoint).", (*designItr)->findElement("StartPoint"));
					CC3DXMLElementList StartPointElement = (*designItr)->getElements("StartPoint");
					ASSERT_OR_THROW("Only one start point can be defined per parallelepiped.", StartPointElement.size() == 1);
					if ((*StartPointElement[0]).findAttribute("x")) { startPos.x = (*StartPointElement[0]).getAttributeAsShort("x"); }
					if ((*StartPointElement[0]).findAttribute("y")) { startPos.y = (*StartPointElement[0]).getAttributeAsShort("y"); }
					if ((*StartPointElement[0]).findAttribute("z")) { startPos.z = (*StartPointElement[0]).getAttributeAsShort("z"); }

					cerr << "      Retrieved start point (" << startPos.x << ", " << startPos.y << ", " << startPos.z << ")" << endl;

					designIsValid = false;

					if ((*designItr)->findElement("EndPoint")) {

						CC3DXMLElementList EndPointElement = (*designItr)->getElements("EndPoint");
						ASSERT_OR_THROW("Only one end point can be defined per parallelepiped.", EndPointElement.size() == 1);
						if ((*EndPointElement[0]).findAttribute("x")) { 
							lenVec[0].x = (*EndPointElement[0]).getAttributeAsShort("x") - startPos.x;
						}
						if ((*EndPointElement[0]).findAttribute("y")) { 
							lenVec[1].y = (*EndPointElement[0]).getAttributeAsShort("y") - startPos.y;
						}
						if ((*EndPointElement[0]).findAttribute("z")) { 
							lenVec[2].z = (*EndPointElement[0]).getAttributeAsShort("z") - startPos.z;
						}

						cerr << "      Retrieved end point (" << lenVec[0].x + startPos.x << ", " << lenVec[1].y + startPos.y << ", " << lenVec[2].z + startPos.z << ")" << endl;

						designIsValid = true;

					}

					if ((*designItr)->findElement("ParaVectors")) {

						if (designIsValid) {
							cerr << "Parallelepiped endpoint already specified. Ignoring vector specifications." << endl;
							continue;
						}

						CC3DXMLElementList ParaVectorsList = (*designItr)->getElements("ParaVectors");

						ASSERT_OR_THROW("Exactly one set of ParaVectors must be defined.", ParaVectorsList.size()==1)
						
						ASSERT_OR_THROW("At least one set of ParaVectors must be defined.", (*ParaVectorsList[0]).findElement("ParaVector"));
						
						CC3DXMLElementList ParaVectorList = (*ParaVectorsList[0]).getElements("ParaVector");

						for (int vectorIndex = 0; vectorIndex < ParaVectorList.size(); ++vectorIndex) {
							if (vectorIndex < lenVec.size()) {
								if (ParaVectorList[vectorIndex]->findAttribute("x")) { lenVec[vectorIndex].x = ParaVectorList[vectorIndex]->getAttributeAsShort("x"); }
								if (ParaVectorList[vectorIndex]->findAttribute("y")) { lenVec[vectorIndex].y = ParaVectorList[vectorIndex]->getAttributeAsShort("y"); }
								if (ParaVectorList[vectorIndex]->findAttribute("z")) { lenVec[vectorIndex].z = ParaVectorList[vectorIndex]->getAttributeAsShort("z"); }

								cerr << "      Retrieved ParaVector (" << lenVec[vectorIndex].x << ", " << lenVec[vectorIndex].y << ", " << lenVec[vectorIndex].z << ")" << endl;

							}
							else { cerr << "      Ignoring parallelepiped vector " << vectorIndex + 1 << endl; }
						}

						designIsValid = true;

					}

					ASSERT_OR_THROW("Parallelepiped not fully defined: prescribe an endpoint (using EndPoint) or set of length vectors (using ParaVectors).", designIsValid);

					ParaDraw(thisQuantityVectorChecked, startPos, lenVec[0], lenVec[1], lenVec[2]);
				}
				else if (thisMethod == "Cylinder") {

					cerr << "   Drawing cylinder." << endl;
					
					Point3D startPos(0, 0, 0);
					Point3D lenVec(-1, -1, -1);

					ASSERT_OR_THROW("One start point must be defined per cylinder (using StartPoint).", (*designItr)->findElement("StartPoint"));
					CC3DXMLElementList StartPointElement = (*designItr)->getElements("StartPoint");
					ASSERT_OR_THROW("Only one start point can be defined per cylinder.", StartPointElement.size() == 1);
					if ((*StartPointElement[0]).findAttribute("x")) { startPos.x = (*StartPointElement[0]).getAttributeAsShort("x"); }
					if ((*StartPointElement[0]).findAttribute("y")) { startPos.y = (*StartPointElement[0]).getAttributeAsShort("y"); }
					if ((*StartPointElement[0]).findAttribute("z")) { startPos.z = (*StartPointElement[0]).getAttributeAsShort("z"); }

					cerr << "      Retrieved start point (" << startPos.x << ", " << startPos.y << ", " << startPos.z << ")" << endl;

					ASSERT_OR_THROW("A radius must be specified per cylinder (using Radius).", (*designItr)->findElement("Radius"));
					CC3DXMLElementList RadiusElement = (*designItr)->getElements("Radius");
					ASSERT_OR_THROW("Only one radius can be defined per cylinder.", RadiusElement.size() == 1);
					short radius = (*RadiusElement[0]).getShort();
					ASSERT_OR_THROW("Cylinder radius must be greater than zero.", radius > 0);

					cerr << "      Retrieved radius " << radius << endl;

					designIsValid = false;

					if ((*designItr)->findElement("EndPoint")) {

						CC3DXMLElementList EndPointElement = (*designItr)->getElements("EndPoint");
						ASSERT_OR_THROW("Only one end point can be defined per cylinder.", EndPointElement.size() == 1);
						if ((*EndPointElement[0]).findAttribute("x")) {
							lenVec.x = (*EndPointElement[0]).getAttributeAsShort("x") - startPos.x;
						}
						if ((*EndPointElement[0]).findAttribute("y")) {
							lenVec.y = (*EndPointElement[0]).getAttributeAsShort("y") - startPos.y;
						}
						if ((*EndPointElement[0]).findAttribute("z")) {
							lenVec.z = (*EndPointElement[0]).getAttributeAsShort("z") - startPos.z;
						}

						cerr << "      Retrieved end point (" << lenVec.x + startPos.x << ", " << lenVec.y + startPos.y << ", " << lenVec.z + startPos.z << ")" << endl;

						designIsValid = true;

					}

					if ((*designItr)->findElement("CylinderVector")) {

						if (designIsValid) {
							cerr << "Cylinder endpoint already specified. Ignoring vector specifications." << endl;
							continue;
						}

						CC3DXMLElementList CylinderVectorList = (*designItr)->getElements("CylinderVector");
						for (int vectorIndex = 0; vectorIndex < CylinderVectorList.size(); ++vectorIndex) {
							if (vectorIndex < 1) {
								if (CylinderVectorList[vectorIndex]->findAttribute("x")) { lenVec.x = CylinderVectorList[vectorIndex]->getAttributeAsShort("x"); }
								if (CylinderVectorList[vectorIndex]->findAttribute("y")) { lenVec.y = CylinderVectorList[vectorIndex]->getAttributeAsShort("y"); }
								if (CylinderVectorList[vectorIndex]->findAttribute("z")) { lenVec.z = CylinderVectorList[vectorIndex]->getAttributeAsShort("z"); }

								cerr << "      Retrieved CylinderVector (" << lenVec.x << ", " << lenVec.y << ", " << lenVec.z << ")" << endl;

							}
							else { cerr << "Ignoring cylinder vector " << vectorIndex + 1 << endl; }
						}

						designIsValid = true;

					}

					ASSERT_OR_THROW("Cylinder not fully defined: prescribe an endpoint (using EndPoint) or a length vector (using CylinderVector).", designIsValid);

					CylinderDraw(thisQuantityVectorChecked, radius, startPos, lenVec);

				}
				else if (thisMethod == "Ellipsoid") {

					cerr << "   Drawing ellipsoid." << endl;
					
					Point3D center = Point3D(0, 0, 0);
					std::vector<short> lenVec(3);
					std::vector<double> angleVec(3, 0.0);
					
					ASSERT_OR_THROW("One center must be defined per ellipsoid (using CenterPoint).", (*designItr)->findElement("CenterPoint"));
					CC3DXMLElementList CenterPointElement = (*designItr)->getElements("CenterPoint");
					ASSERT_OR_THROW("Only one focus can be defined per ellipsoid.", CenterPointElement.size() == 1);
					if ((*CenterPointElement[0]).findAttribute("x")) { center.x = (*CenterPointElement[0]).getAttributeAsShort("x"); }
					if ((*CenterPointElement[0]).findAttribute("y")) { center.y = (*CenterPointElement[0]).getAttributeAsShort("y"); }
					if ((*CenterPointElement[0]).findAttribute("z")) { center.z = (*CenterPointElement[0]).getAttributeAsShort("z"); }

					cerr << "      Retrieved center point (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;

					ASSERT_OR_THROW("One set of ellipsoid half lengths must be specified per ellipsoid (using EllipsoidHalfLengths)", (*designItr)->findElement("EllipsoidHalfLengths"));
					CC3DXMLElementList EllipsoidHalfLengthsList = (*designItr)->getElements("EllipsoidHalfLengths");
					ASSERT_OR_THROW("Exactly one set of ellipsoid half lengths must be defined per ellipsoid (using EllipsoidHalfLengths).", EllipsoidHalfLengthsList.size() == 1);
					ASSERT_OR_THROW("An unrotated ellipsoid half length must be specified along the X-direction", EllipsoidHalfLengthsList[0]->findElement("HalfLengthX"));
					ASSERT_OR_THROW("An unrotated ellipsoid half length must be specified along the Y-direction", EllipsoidHalfLengthsList[0]->findElement("HalfLengthY"));
					ASSERT_OR_THROW("An unrotated ellipsoid half length must be specified along the Z-direction", EllipsoidHalfLengthsList[0]->findElement("HalfLengthZ"));
					CC3DXMLElementList EllipsoidHalfLength;
					EllipsoidHalfLength = EllipsoidHalfLengthsList[0]->getElements("HalfLengthX");
					ASSERT_OR_THROW("Exactly one unrotated ellipsoid half length must be specified along the X-direction", EllipsoidHalfLength.size() == 1);
					lenVec[0] = EllipsoidHalfLength[0]->getShort();
					EllipsoidHalfLength = EllipsoidHalfLengthsList[0]->getElements("HalfLengthY");
					ASSERT_OR_THROW("Exactly one unrotated ellipsoid half length must be specified along the Y-direction", EllipsoidHalfLength.size() == 1);
					lenVec[1] = EllipsoidHalfLength[0]->getShort();
					EllipsoidHalfLength = EllipsoidHalfLengthsList[0]->getElements("HalfLengthZ");
					ASSERT_OR_THROW("Exactly one unrotated ellipsoid half length must be specified along the Z-direction", EllipsoidHalfLength.size() == 1);
					lenVec[2] = EllipsoidHalfLength[0]->getShort();

					cerr << "      Retrieved lengths " << lenVec[0] << ", " << lenVec[1] << ", " << lenVec[2] << endl;
					
					if ((*designItr)->findElement("EllipsoidAngles")) {

						CC3DXMLElementList EllipsoidAnglesList = (*designItr)->getElements("EllipsoidAngles");
						ASSERT_OR_THROW("Only one set of ellipsoid half lengths can be defined per ellipsoid (using EllipsoidAngles).", EllipsoidAnglesList.size() == 1);

						CC3DXMLElementList EllipsoidAngle;
						if (EllipsoidAnglesList[0]->findElement("AngleX")) {
							EllipsoidAngle = EllipsoidAnglesList[0]->getElements("AngleX");
							ASSERT_OR_THROW("Only one ellipsoid rotation can be specified about the X-direction", EllipsoidAngle.size() == 1);
							angleVec[0] = EllipsoidAngle[0]->getDouble();
							cerr << "      Retrieved X-angle " << angleVec[0] << endl;
						}
						if (EllipsoidAnglesList[0]->findElement("AngleY")) {
							EllipsoidAngle = EllipsoidAnglesList[0]->getElements("AngleY");
							ASSERT_OR_THROW("Only one ellipsoid rotation can be specified about the Y-direction", EllipsoidAngle.size() == 1);
							angleVec[1] = EllipsoidAngle[0]->getDouble();
							cerr << "      Retrieved Y-angle " << angleVec[1] << endl;
						}
						if (EllipsoidAnglesList[0]->findElement("AngleZ")) {
							EllipsoidAngle = EllipsoidAnglesList[0]->getElements("AngleZ");
							ASSERT_OR_THROW("Only one ellipsoid rotation can be specified about the Z-direction", EllipsoidAngle.size() == 1);
							angleVec[2] = EllipsoidAngle[0]->getDouble();
							cerr << "      Retrieved Z-angle " << angleVec[2] << endl;
						}

					}
					
					EllipsoidDraw(thisQuantityVectorChecked, center, lenVec, angleVec);

				}
				else if (thisMethod == "Sphere") {

					cerr << "   Drawing sphere." << endl;

					Point3D center = Point3D(0, 0, 0);
					short radius;

					ASSERT_OR_THROW("One center must be defined per sphere (using CenterPoint).", (*designItr)->findElement("CenterPoint"));
					CC3DXMLElementList CenterPointElement = (*designItr)->getElements("CenterPoint");
					ASSERT_OR_THROW("Only one focus can be defined per sphere.", CenterPointElement.size() == 1);
					if ((*CenterPointElement[0]).findAttribute("x")) { center.x = (*CenterPointElement[0]).getAttributeAsShort("x"); }
					if ((*CenterPointElement[0]).findAttribute("y")) { center.y = (*CenterPointElement[0]).getAttributeAsShort("y"); }
					if ((*CenterPointElement[0]).findAttribute("z")) { center.z = (*CenterPointElement[0]).getAttributeAsShort("z"); }

					cerr << "      Retrieved center point (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;

					ASSERT_OR_THROW("One radius must be specified per sphere (using Radius)", (*designItr)->findElement("Radius"));
					CC3DXMLElementList RadiusList = (*designItr)->getElements("Radius");
					ASSERT_OR_THROW("Exactly radius must be defined per sphere (using Radius).", RadiusList.size() == 1);
					radius = RadiusList[0]->getShort();
					
					cerr << "      Retrieved radius " << radius << endl;

					EllipsoidDraw(thisQuantityVectorChecked, center, std::vector<short>(3, radius), std::vector<double>(3, 0.0));

				}

			}

		}

	}

	pUtils->unsetLock(lockPtr);
	
}

void NCMaterialsPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);

}

void NCMaterialsPlugin::handleEvent(CC3DEvent & _event) {
	if (_event.id == LATTICE_RESIZE) {

		pUtils->setLock(lockPtr);
		
		WatchableField3D<NCMaterialsData *> NCMaterialsFieldOld = WatchableField3D<NCMaterialsData *>(fieldDim, 0);
		for (int z = 0; z < fieldDim.z; ++z)
			for (int y = 0; y < fieldDim.y; ++y)
				for (int x = 0; x < fieldDim.x; ++x) {
					Point3D pt = Point3D(x, y, z);
					NCMaterialsFieldOld.set(pt, NCMaterialsField->get(pt));
				}

		CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);
		fieldDim = ev.newDim;

		initializeNCMaterialsField(fieldDim, true);

		for (int z = 0; z < fieldDim.z; ++z)
			for (int y = 0; y < fieldDim.y; ++y)
				for (int x = 0; x < fieldDim.x; ++x) {
					Point3D pt = Point3D(x, y, z);
					Point3D ptNew = pt;
					ptNew.x += ev.shiftVec.x;
					ptNew.y += ev.shiftVec.y;
					ptNew.z += ev.shiftVec.z;
					NCMaterialsField->get(ptNew)->setNCMaterialsQuantityVec(NCMaterialsFieldOld.get(pt)->getNCMaterialsQuantityVec());
				}

		pUtils->unsetLock(lockPtr);

	}
}


double NCMaterialsPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    
	if (!NCMaterialsInitialized) {
        pUtils->setLock(lockPtr);
        initializeNCMaterials();
        pUtils->unsetLock(lockPtr);
    }

    double energy = 0;
    double distance = 0;
    Neighbor neighbor;
    vector<float> targetQuantityVec(numberOfMaterials);

    // Target medium and durability
    if (oldCell == 0) {
        targetQuantityVec = NCMaterialsField->get(pt)->NCMaterialsQuantityVec;
        energy += NCMaterialDurabilityEnergy(targetQuantityVec);
    }
    vector<float> copyQuantityVector(numberOfMaterials);
    if ((newCell == 0) && (oldCell != 0)) { copyQuantityVector = calculateCopyQuantityVec(oldCell, pt); }

    CellG *nCell = 0;

	unsigned int nIdx;
	std::vector<float> nQuantityVec;

    if (weightDistance) {
        for (nIdx = 0; nIdx <= maxNeighborIndexAdh; ++nIdx) {
            neighbor = boundaryStrategyAdh->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

            if (!neighbor.distance) continue;

            distance = neighbor.distance;

            nCell = potts->getCellFieldG()->get(neighbor.pt);
			nQuantityVec = NCMaterialsField->get(neighbor.pt)->NCMaterialsQuantityVec;

            if (nCell != oldCell) {
                if (nCell == 0) energy -= NCMaterialContactEnergy(oldCell, nQuantityVec) / neighbor.distance;
                else if (oldCell == 0) energy -= NCMaterialContactEnergy(nCell, targetQuantityVec) / neighbor.distance;
            }
            if (nCell != newCell) {
                if (nCell == 0) energy += NCMaterialContactEnergy(newCell, nQuantityVec) / neighbor.distance;
                else if (newCell == 0) energy += NCMaterialContactEnergy(nCell, copyQuantityVector) / neighbor.distance;
            }
        }
    }
    else {
        //default behaviour  no energy weighting

        for (nIdx = 0; nIdx <= maxNeighborIndexAdh; ++nIdx) {
            neighbor = boundaryStrategyAdh->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

            if (!neighbor.distance) continue;

            nCell = potts->getCellFieldG()->get(neighbor.pt);
			nQuantityVec = NCMaterialsField->get(neighbor.pt)->NCMaterialsQuantityVec;

            if (nCell != oldCell) {
                if (nCell == 0) energy -= NCMaterialContactEnergy(oldCell, nQuantityVec);
                else if (oldCell == 0) energy -= NCMaterialContactEnergy(nCell, targetQuantityVec);
            }
            if (nCell != newCell) {
                if (nCell == 0) energy += NCMaterialContactEnergy(newCell, nQuantityVec);
                else if (newCell == 0) energy += NCMaterialContactEnergy(nCell, copyQuantityVector);
            }
        }
    }

    return energy;
}

double NCMaterialsPlugin::NCMaterialContactEnergy(const CellG *cell, std::vector<float> _qtyVec) {

    double energy = 0.0;
	std::vector<float> AdhesionCoefficients;
    if (cell != 0) {
		AdhesionCoefficients = NCMaterialCellDataAccessor.get(cell->extraAttribPtr)->AdhesionCoefficients;
        for (int i = 0; i < AdhesionCoefficients.size() ; ++i) energy += AdhesionCoefficients[i] * _qtyVec[i];
    }

    return energy;

}

double NCMaterialsPlugin::NCMaterialDurabilityEnergy(std::vector<float> _qtyVec) {

    double energy = 0.0;
    float thisEnergy;

    for (int i = 0; i < _qtyVec.size(); ++i) {
        thisEnergy = _qtyVec[i]* NCMaterialsVec[i].getDurabilityLM();
		if (thisEnergy > 0.0) { energy += (double)thisEnergy; }
    }

    return energy;

}

void NCMaterialsPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    // If source agent is a cell and target agent is the medium, then target NC materials are removed
    // If source agent is the medium, materials advect

	NCMaterialsData *NCMaterialsDataLocal = NCMaterialsField->get(pt);
	std::vector<float> & NCMaterialsQuantityVec = NCMaterialsDataLocal->NCMaterialsQuantityVec;
	std::vector<float> NCMaterialsQuantityVecNew(numberOfMaterials);

    if (newCell) { // Source agent is a cell
        if (!oldCell){
			NCMaterialsQuantityVec = NCMaterialsQuantityVecNew;
			NCMaterialsField->set(pt, NCMaterialsDataLocal);
        }
    }
    else { // Source agent is a medium
        if (oldCell){
			NCMaterialsQuantityVec = calculateCopyQuantityVec(oldCell, pt);
			NCMaterialsField->set(pt, NCMaterialsDataLocal);
        }
    }

}

std::vector<float> NCMaterialsPlugin::calculateCopyQuantityVec(const CellG * _cell, const Point3D &pt) {

    std::vector<float> copyQuantityVec(numberOfMaterials);

    // Calculate copy quantity vector
    // quantity vector is mean of all transferable neighborhood components + target cell remodeling quantity
    CellG *neighborCell;
	if (_cell) { copyQuantityVec = NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity; }

    float numberOfMediumNeighbors = 1.0;
    std::vector<float> neighborQuantityVector(numberOfMaterials);
    std::vector<Neighbor> neighbors = getFirstOrderNeighbors(pt);
    WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    for (int nIdx = 0; nIdx < neighbors.size(); ++nIdx){
        neighborCell = fieldG->get(neighbors[nIdx].pt);
        if (neighborCell) {continue;}
		neighborQuantityVector = NCMaterialsField->get(neighbors[nIdx].pt)->NCMaterialsQuantityVec;
		for (int i = 0; i < NCMaterialsVec.size(); ++i) {
			if ( !(NCMaterialsVec[i].getTransferable()) ) { neighborQuantityVector[i] = 0.0; }
		}

		for (int i = 0; i < copyQuantityVec.size(); ++i) { copyQuantityVec[i] += neighborQuantityVector[i]; }
        numberOfMediumNeighbors += 1.0;
    }

	for (std::vector<float>::iterator i = copyQuantityVec.begin(); i != copyQuantityVec.end(); ++i) { *i /= numberOfMediumNeighbors; }

    std::vector<float> copyQuantityVecChecked = NCMaterialsPlugin::checkQuantities(copyQuantityVec);

	return copyQuantityVecChecked;

}

void NCMaterialsPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
        set<unsigned char> cellTypesSet;

}

void NCMaterialsPlugin::initializeNCMaterialsField(Dim3D _fieldDim, bool _resetting) {
	if (_resetting) deleteNCMaterialsField();

	NCMaterialsField = new WatchableField3D<NCMaterialsData *>(fieldDim, 0);

	// initialize NC material quantity vector field values
	// default is all first NCM component

	cerr << "Initializing NCMaterials quantity field values..." << endl;

	Point3D pt;
	NCMaterialsData *NCMaterialsDataLocal;
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				pt = Point3D(x, y, z);

				NCMaterialsDataLocal = new NCMaterialsData();
				std::vector<float> & NCMaterialsQuantityVec = NCMaterialsDataLocal->NCMaterialsQuantityVec;
				NCMaterialsDataLocal->numMtls = numberOfMaterials;
				std::vector<float> NCMaterialsQuantityVecNew(numberOfMaterials);
				NCMaterialsQuantityVecNew[0] = 1.0;
				NCMaterialsQuantityVec = NCMaterialsQuantityVecNew;
				NCMaterialsField->set(pt, NCMaterialsDataLocal);
			}
}

void NCMaterialsPlugin::deleteNCMaterialsField() {
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				Point3D pt = Point3D(x, y, z);
				NCMaterialsData *d = NCMaterialsField->get(pt);
				delete d;
			}

	delete NCMaterialsField;
	NCMaterialsField = 0;
}

void NCMaterialsPlugin::initializeNCMaterials() {
    
	cerr << "NCMaterials plugin initialization begin: " << NCMaterialsInitialized << endl;

	if (NCMaterialsInitialized)//we double-check this flag to makes sure this function does not get called multiple times by different threads
        return;


	// initialize cell type->remodeling quantity map
	//      first collect inputs as a 2D array by type and material ids, then assemble map

	cerr << "Initializing remodeling quantities..." << endl;

	int thisTypeId;
	int maxTypeId = (int)automaton->getMaxTypeId();
	int NCMaterialIdx;
	double thisRemodelingQuantity;
	std::string NCMaterialName;
	std::string cellTypeName;
	std::vector<std::string> cellTypeNamesByTypeId;
	std::vector<std::vector<float> > RemodelingQuantityByTypeId(maxTypeId + 1, vector<float>(numberOfMaterials));
	typeToRemodelingQuantityMap.clear();

	for (int i = 0; i < NCMaterialRemodelingQuantityXMLVec.size(); ++i) {
		cellTypeName = NCMaterialRemodelingQuantityXMLVec[i]->getAttribute("CellType");
		NCMaterialName = NCMaterialRemodelingQuantityXMLVec[i]->getAttribute("Material");
		thisRemodelingQuantity = NCMaterialRemodelingQuantityXMLVec[i]->getDouble();

		cerr << "   Initializing (" << cellTypeName << ", " << NCMaterialName << " ): " << thisRemodelingQuantity << endl;

		thisTypeId = (int)automaton->getTypeId(cellTypeName);
		NCMaterialIdx = getNCMaterialIndexByName(NCMaterialName);
		cellTypeNamesByTypeId.push_back(cellTypeName);
		RemodelingQuantityByTypeId[thisTypeId][NCMaterialIdx] = (float)thisRemodelingQuantity;
	}
	//      Make the cell type->remodeling quantity map
	std::vector<float> thisRemodelingQuantityVec;
	for (int i = 0; i < cellTypeNamesByTypeId.size(); ++i) {
		cellTypeName = cellTypeNamesByTypeId[i];
		thisTypeId = (int)automaton->getTypeId(cellTypeName);

		cerr << "   Setting remodeling quantity for cell type " << thisTypeId << ": " << cellTypeName << endl;

		thisRemodelingQuantityVec = RemodelingQuantityByTypeId[thisTypeId];
		typeToRemodelingQuantityMap.insert(make_pair(cellTypeName, thisRemodelingQuantityVec));
	}

	// initialize cell adhesion coefficients by cell type from user specification

	cerr << "Initializing cell-NCMaterial interface adhesion coefficients by cell type and material component..." << endl;

	AdhesionCoefficientsByTypeId.clear();
	std::vector<std::vector<float> > AdhesionCoefficientsByTypeId(maxTypeId + 1, std::vector<float>(numberOfMaterials));
	double thisAdhesionCoefficient;
	for (int i = 0; i < NCMaterialAdhesionXMLVec.size(); ++i) {
		cellTypeName = NCMaterialAdhesionXMLVec[i]->getAttribute("CellType");
		NCMaterialName = NCMaterialAdhesionXMLVec[i]->getAttribute("Material");
		thisAdhesionCoefficient = NCMaterialAdhesionXMLVec[i]->getDouble();

		cerr << "   Initializing (" << cellTypeName << ", " << NCMaterialName << " ): " << thisAdhesionCoefficient << endl;

		thisTypeId = (int)automaton->getTypeId(cellTypeName);
		NCMaterialIdx = getNCMaterialIndexByName(NCMaterialName);
		AdhesionCoefficientsByTypeId[thisTypeId][NCMaterialIdx] = (float)thisAdhesionCoefficient;
	}

	// assign remodeling quantities and adhesion coefficients to cells by type and material name from user specification

	cerr << "Assigning remodeling quantities and adhesion coefficients to cells..." << endl;

	std::map<std::string, std::vector<float> >::iterator mitr;
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	CellInventory * cellInventoryPtr = &potts->getCellInventory();
	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
		cell = cellInventoryPtr->getCell(cInvItr);
		std::vector<float> & RemodelingQuantity = NCMaterialCellDataAccessor.get(cell->extraAttribPtr)->RemodelingQuantity;
		std::vector<float> & AdhesionCoefficients = NCMaterialCellDataAccessor.get(cell->extraAttribPtr)->AdhesionCoefficients;

		std::vector<float> RemodelingQuantityNew(numberOfMaterials);
		RemodelingQuantity = RemodelingQuantityNew;
		RemodelingQuantity[0] = 1.0;

		cellTypeName = automaton->getTypeName(automaton->getCellType(cell));
		mitr = typeToRemodelingQuantityMap.find(cellTypeName);
		if (mitr != typeToRemodelingQuantityMap.end()) {
			RemodelingQuantity = mitr->second;
		}

		AdhesionCoefficients = AdhesionCoefficientsByTypeId[(int)cell->type];
	}

    NCMaterialsInitialized = true;

	cerr << "NCMaterials plugin initialization complete: " << NCMaterialsInitialized << endl;

}

void NCMaterialsPlugin::setMaterialNameVector() {
	fieldNameVec.clear();
	for (std::map<std::string, int>::iterator mitr = NCMaterialNameIndexMap.begin(); mitr != NCMaterialNameIndexMap.end(); ++mitr)
		fieldNameVec.push_back(mitr->first);
}

std::vector<float> NCMaterialsPlugin::checkQuantities(std::vector<float> _qtyVec) {
	float qtySum = 0.0;
	for (int i = 0; i < _qtyVec.size(); ++i) {
		if (_qtyVec[i] < 0.0 || isnan(_qtyVec[i])) { _qtyVec[i] = 0.0; }
		else if (_qtyVec[i] > 1.0) { _qtyVec[i] = 1.0; }
		qtySum += _qtyVec[i];
    }
	if (qtySum > 1.0) { for (int i = 0; i < _qtyVec.size(); ++i) { _qtyVec[i] /= qtySum; } }
    return _qtyVec;
}

float NCMaterialsPlugin::getLocalDiffusivity(const Point3D &pt, std::string _fieldName) {
	
	if (potts->getCellFieldG()->get(pt)) return 0.0;

	float diffCoeff = 0.0;
	std::vector<float> qtyVec = this->getMediumNCMaterialQuantityVector(pt);

	for (int i = 0; i < this->numberOfMaterials; ++i)
		diffCoeff += this->NCMaterialsVec[i].getFieldDiffusivity(_fieldName) * qtyVec[i];

	return diffCoeff;
}

void NCMaterialsPlugin::setRemodelingQuantityByName(const CellG * _cell, std::string _NCMaterialName, float _quantity) {
    int _idx = getNCMaterialIndexByName(_NCMaterialName);
    setRemodelingQuantityByIndex(_cell, _idx, _quantity);
}

void NCMaterialsPlugin::setRemodelingQuantityByIndex(const CellG * _cell, int _idx, float _quantity) {
	std::vector<float> & RemodelingQuantity = NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	if (_idx >= 0 && _idx < RemodelingQuantity.size()) { RemodelingQuantity[_idx] = _quantity; }
}

void NCMaterialsPlugin::setRemodelingQuantityVector(const CellG * _cell, std::vector<float> _quantityVec) {
	std::vector<float> & RemodelingQuantity = NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	RemodelingQuantity = _quantityVec;
}

void NCMaterialsPlugin::assignNewRemodelingQuantityVector(const CellG * _cell, int _numMtls) {
	int numMtls;
	if (_numMtls < 0) numMtls = numberOfMaterials;
	else numMtls = _numMtls;
	std::vector<float> RemodelingQuantityNew(numMtls);
	std::vector<float> & RemodelingQuantity = NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	RemodelingQuantity = RemodelingQuantityNew;
}

void NCMaterialsPlugin::setMediumNCMaterialQuantityByName(const Point3D &pt, std::string _NCMaterialName, float _quantity) {
    setMediumNCMaterialQuantityByIndex(pt, getNCMaterialIndexByName(_NCMaterialName), _quantity);
}

void NCMaterialsPlugin::setMediumNCMaterialQuantityByIndex(const Point3D &pt, int _idx, float _quantity) {
    NCMaterialsData *NCMaterialsDataLocal = NCMaterialsField->get(pt);
	std::vector<float> & NCMaterialsQuantityVec = NCMaterialsDataLocal->NCMaterialsQuantityVec;
	NCMaterialsQuantityVec[_idx] = _quantity;
    NCMaterialsField->set(pt, NCMaterialsDataLocal);
}

void NCMaterialsPlugin::setMediumNCMaterialQuantityVector(const Point3D &pt, std::vector<float> _quantityVec) {
    NCMaterialsData *NCMaterialsDataLocal = NCMaterialsField->get(pt);
	std::vector<float> & NCMaterialsQuantityVec = NCMaterialsDataLocal->NCMaterialsQuantityVec;
	NCMaterialsQuantityVec = _quantityVec;
    NCMaterialsField->set(pt, NCMaterialsDataLocal);
}

void NCMaterialsPlugin::assignNewMediumNCMaterialQuantityVector(const Point3D &pt, int _numMtls) {
	int numMtls;
	if (_numMtls < 0) numMtls = numberOfMaterials;
	else numMtls = _numMtls;
    NCMaterialsData *NCMaterialsDataLocal = NCMaterialsField->get(pt);
	std::vector<float> NCMaterialsQuantityVecNew(numMtls);
	std::vector<float> & NCMaterialsQuantityVec = NCMaterialsDataLocal->NCMaterialsQuantityVec;
	NCMaterialsDataLocal->numMtls = numMtls;
	NCMaterialsQuantityVec = NCMaterialsQuantityVecNew;
	NCMaterialsField->set(pt, NCMaterialsDataLocal);
}

void NCMaterialsPlugin::setNCMaterialDurabilityByName(std::string _NCMaterialName, float _durabilityLM) {
    setNCMaterialDurabilityByIndex(getNCMaterialIndexByName(_NCMaterialName), _durabilityLM);
}

void NCMaterialsPlugin::setNCMaterialDurabilityByIndex(int _idx, float _durabilityLM) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx < NCMaterialsVec.size() ) NCMaterialsVec[_idx].setDurabilityLM(_durabilityLM);
}

void NCMaterialsPlugin::setNCMaterialAdvectingByName(std::string _NCMaterialName, bool _isAdvecting) {
    setNCMaterialAdvectingByIndex(getNCMaterialIndexByName(_NCMaterialName), _isAdvecting);
}

void NCMaterialsPlugin::setNCMaterialAdvectingByIndex(int _idx, bool _isAdvecting) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx < NCMaterialsVec.size() ) NCMaterialsVec[_idx].setTransferable(_isAdvecting);
}

void NCMaterialsPlugin::setNCAdhesionByCell(const CellG *_cell, std::vector<float> _adhVec) {
    vector<float> & AdhesionCoefficients = NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->AdhesionCoefficients;
	AdhesionCoefficients = _adhVec;
}

void NCMaterialsPlugin::setNCAdhesionByCellAndMaterialIndex(const CellG *_cell, int _idx, float _val) {
	vector<float> & AdhesionCoefficients = NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->AdhesionCoefficients;
	if (_idx >= 0 && _idx < AdhesionCoefficients.size()) AdhesionCoefficients[_idx] = _val;
}

void NCMaterialsPlugin::setNCAdhesionByCellAndMaterialName(const CellG *_cell, std::string _NCMaterialName, float _val) {
	setNCAdhesionByCellAndMaterialIndex(_cell, getNCMaterialIndexByName(_NCMaterialName), _val);
}

float NCMaterialsPlugin::getRemodelingQuantityByName(const CellG * _cell, std::string _NCMaterialName) {
    return getRemodelingQuantityByIndex(_cell, getNCMaterialIndexByName(_NCMaterialName));
}

float NCMaterialsPlugin::getRemodelingQuantityByIndex(const CellG * _cell, int _idx) {
	std::vector<float> remodelingQuantityVector = getRemodelingQuantityVector(_cell);
	if (_idx < 0 || _idx > remodelingQuantityVector.size() - 1 ) {
        ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!" , false);
        return -1.0;
    }
	return remodelingQuantityVector[_idx];
}

std::vector<float> NCMaterialsPlugin::getRemodelingQuantityVector(const CellG * _cell) {
    if (_cell) return NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	else return std::vector<float>(numberOfMaterials, 0.0);
}

float NCMaterialsPlugin::getMediumNCMaterialQuantityByName(const Point3D &pt, std::string _NCMaterialName) {
    return getMediumNCMaterialQuantityByIndex(pt, getNCMaterialIndexByName(_NCMaterialName));
}

float NCMaterialsPlugin::getMediumNCMaterialQuantityByIndex(const Point3D &pt, int _idx) {
	std::vector<float> NCMaterialsQuantityVec = getMediumNCMaterialQuantityVector(pt);
	return NCMaterialsQuantityVec[_idx];
}

std::vector<float> NCMaterialsPlugin::getMediumNCMaterialQuantityVector(const Point3D &pt) {
	return NCMaterialsField->get(pt)->NCMaterialsQuantityVec;
}

std::vector<float> NCMaterialsPlugin::getMediumAdvectingNCMaterialQuantityVector(const Point3D &pt) {
    std::vector<float> _qtyVec = getMediumNCMaterialQuantityVector(pt);
    std::vector<NCMaterialComponentData> _ncmVec = getNCMaterialsVec();

	for (int i = 0; i < _qtyVec.size(); ++i) { if (!_ncmVec[i].getTransferable()) { _qtyVec[i] = 0.0; } }
    
	return _qtyVec;
}

float NCMaterialsPlugin::getNCMaterialDurabilityByName(std::string _NCMaterialName) {
    return getNCMaterialDurabilityByIndex(getNCMaterialIndexByName(_NCMaterialName));
}

float NCMaterialsPlugin::getNCMaterialDurabilityByIndex(int _idx) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx < NCMaterialsVec.size()) return NCMaterialsVec[_idx].getDurabilityLM();
	ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!", false);
    return 0;
}

bool NCMaterialsPlugin::getNCMaterialAdvectingByName(std::string _NCMaterialName) {
    return getNCMaterialAdvectingByIndex(getNCMaterialIndexByName(_NCMaterialName));
}

bool NCMaterialsPlugin::getNCMaterialAdvectingByIndex(int _idx) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx > NCMaterialsVec.size() - 1 ) return NCMaterialsVec[_idx].getTransferable();
	ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!", false);
    return 0;
}

int NCMaterialsPlugin::getNCMaterialIndexByName(std::string _NCMaterialName){
	std::map<std::string, int>::iterator mitr = NCMaterialNameIndexMap.find(_NCMaterialName);
    if ( mitr != NCMaterialNameIndexMap.end() ) return mitr->second;
    return -1;
}

std::vector<float> NCMaterialsPlugin::getNCAdhesionByCell(const CellG *_cell) {
    return NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->AdhesionCoefficients;
}

std::vector<float> NCMaterialsPlugin::getNCAdhesionByCellTypeId(int _idx) {
    std::vector<float> _adhVec(numberOfMaterials);
    if (_idx < 0 || _idx > AdhesionCoefficientsByTypeId.size() - 1) {ASSERT_OR_THROW("Material index out of range!" , false);}
    else return AdhesionCoefficientsByTypeId[_idx];
}

float NCMaterialsPlugin::getNCAdhesionByCellAndMaterialIndex(const CellG *_cell, int _idx) {
	return NCMaterialCellDataAccessor.get(_cell->extraAttribPtr)->AdhesionCoefficients[_idx];
}

float NCMaterialsPlugin::getNCAdhesionByCellAndMaterialName(const CellG *_cell, std::string _NCMaterialName) {
	return getNCAdhesionByCellAndMaterialIndex(_cell, getNCMaterialIndexByName(_NCMaterialName));
}

std::vector<Neighbor> NCMaterialsPlugin::getFirstOrderNeighbors(const Point3D &pt) {
    // initialize neighborhood according to Potts neighborhood
    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
    std::vector<Neighbor> neighbors;
	Neighbor neighbor;
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
		neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
		if (!neighbor.distance) continue;
        neighbors.push_back(neighbor);
    }
    return neighbors;
}

float NCMaterialsPlugin::calculateTotalInterfaceQuantityByMaterialIndex(CellG *cell, int _materialIdx) {
	return ncMaterialsSteppable->calculateTotalInterfaceQuantityByMaterialIndex(cell, _materialIdx);
}
float NCMaterialsPlugin::calculateTotalInterfaceQuantityByMaterialName(CellG *cell, std::string _materialName) {
	return ncMaterialsSteppable->calculateTotalInterfaceQuantityByMaterialName(cell, _materialName);
}

float NCMaterialsPlugin::calculateCellProbabilityProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) { 
	return ncMaterialsSteppable->calculateCellProbabilityProliferation(cell, _ncmaterialsField); 
}

float NCMaterialsPlugin::calculateCellProbabilityDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) { 
	return ncMaterialsSteppable->calculateCellProbabilityDeath(cell, _ncmaterialsField); 
}

float NCMaterialsPlugin::calculateCellProbabilityDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return ncMaterialsSteppable->calculateCellProbabilityDifferentiation(cell, newCellType, _ncmaterialsField); 
}

float NCMaterialsPlugin::calculateCellProbabilityAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return ncMaterialsSteppable->calculateCellProbabilityAsymmetricDivision(cell, newCellType, _ncmaterialsField); 
}

bool NCMaterialsPlugin::getCellResponseProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return ncMaterialsSteppable->getCellResponseProliferation(cell, _ncmaterialsField);
}
bool NCMaterialsPlugin::getCellResponseDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return ncMaterialsSteppable->getCellResponseDeath(cell, _ncmaterialsField);
}
bool NCMaterialsPlugin::getCellResponseDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return ncMaterialsSteppable->getCellResponseDifferentiation(cell, newCellType, _ncmaterialsField);
}
bool NCMaterialsPlugin::getCellResponseAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField) {
	return ncMaterialsSteppable->getCellResponseAsymmetricDivision(cell, newCellType, _ncmaterialsField);
}

void NCMaterialsPlugin::ParaDraw(std::vector<float> _qtyVec, Point3D _startPos, Point3D _lenVec1, Point3D _lenVec2, Point3D _lenVec3) {

	std::vector<float> qtyVec = checkQuantities(_qtyVec);
	
	Point3D StartPos = _startPos;
	Point3D lenVec1 = _lenVec1;
	Point3D lenVec2 = _lenVec2;
	Point3D lenVec3 = _lenVec3;
	if (_lenVec1.x == -1) { lenVec1.x = fieldDim.x; };
	if (_lenVec1.y == -1) { lenVec1.y = fieldDim.y; };
	if (_lenVec1.z == -1) { lenVec1.z = fieldDim.z; };
	if (_lenVec2.x == -1) { lenVec2.x = fieldDim.x; };
	if (_lenVec2.y == -1) { lenVec2.y = fieldDim.y; };
	if (_lenVec2.z == -1) { lenVec2.z = fieldDim.z; };
	if (_lenVec3.x == -1) { lenVec3.x = fieldDim.x; };
	if (_lenVec3.y == -1) { lenVec3.y = fieldDim.y; };
	if (_lenVec3.z == -1) { lenVec3.z = fieldDim.z; };

	int len12 = dotProduct(lenVec1, lenVec1);
	int len22 = dotProduct(lenVec2, lenVec2);
	int len32 = dotProduct(lenVec3, lenVec3);

	cerr << "*********** ParaDraw! ***********" << endl;
	cerr << "   Drawing with quantity vector (";
	for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
		cerr << qtyVec[materialIndex];
		if (materialIndex < numberOfMaterials - 1) { cerr << ", "; }
		else { cerr << ")" << endl; }
	}
	cerr << "   Start point (" << StartPos.x << ", " << StartPos.y << ", " << StartPos.z << ")" << endl;
	cerr << "   Parallelepiped squared lengths " << len12 << ", " << len22 << ", " << len32 << endl;
	cerr << "   Drawing... ";

	Point3D pt(0, 0, 0);
	Point3D ptRel(0, 0, 0);
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				pt = Point3D(x, y, z);
				ptRel = pt - StartPos;
				
				int dotProductPt1 = dotProduct(ptRel, lenVec1);
				if ( (dotProductPt1 < 0) || (dotProductPt1 > len12) ) continue;
				int dotProductPt2 = dotProduct(ptRel, lenVec2);
				if ( (dotProductPt2 < 0) || (dotProductPt2 > len22) ) continue;
				int dotProductPt3 = dotProduct(ptRel, lenVec3);
				if ( (dotProductPt3 < 0) || (dotProductPt3 > len32) ) continue;

				setMediumNCMaterialQuantityVector(pt, qtyVec);

			}

	cerr << "complete." << endl;
	cerr << "*********************************" << endl;

}

void NCMaterialsPlugin::CylinderDraw(std::vector<float> _qtyVec, short _radius, Point3D _startPos, Point3D _lenVec) {

	std::vector<float> qtyVec = checkQuantities(_qtyVec);
	
	int radius = _radius;
	Point3D startPos = _startPos;
	Point3D lenVec = _lenVec;
	if (_lenVec.x == -1) { lenVec.x = fieldDim.x; };
	if (_lenVec.y == -1) { lenVec.y = fieldDim.y; };
	if (_lenVec.z == -1) { lenVec.z = fieldDim.z; };

	cerr << "********* CylinderDraw! *********" << endl;
	cerr << "   Drawing with quantity vector (";
	for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
		cerr << qtyVec[materialIndex];
		if (materialIndex < numberOfMaterials - 1) { cerr << ", "; }
		else { cerr << ")" << endl; }
	}
	
	int magLenVec2 = dotProduct(lenVec, lenVec);
	int scaledRadius2 = magLenVec2*radius*radius;

	cerr << "   Cylinder radius " << radius << endl;
	cerr << "   Start point (" << startPos.x << ", " << startPos.y << ", " << startPos.z << ")" << endl;
	cerr << "   Lengths " << lenVec.x << ", " << lenVec.y << ", " << lenVec.z << endl;
	cerr << "   Cylinder squared length " << magLenVec2 << endl;
	cerr << "   Drawing... ";

	Point3D pt(0, 0, 0);
	Point3D ptRel(0, 0, 0);
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				pt = Point3D(x, y, z);
				ptRel = pt - startPos;

				// axial test

				int axialComp = dotProduct(ptRel, lenVec);
				if (axialComp < 0 || axialComp > magLenVec2) continue;

				// radial test

				int totalMag = magLenVec2*dotProduct(ptRel, ptRel);
				int radialLength = totalMag - axialComp*axialComp;
				if (radialLength > scaledRadius2) continue;
				
				setMediumNCMaterialQuantityVector(pt, qtyVec);

			}

	cerr << "complete." << endl;
	cerr << "*********************************" << endl;

}

void NCMaterialsPlugin::EllipsoidDraw(std::vector<float> _qtyVec, Point3D _center, std::vector<short> _lenVec, std::vector<double> _angleVec) {

	#define PI 3.14159265

	std::vector<float> qtyVec = checkQuantities(_qtyVec);

	Point3D center = _center;
	std::vector<short> lenVec = _lenVec;
	std::vector<double> angleVec = _angleVec;

	cerr << "********* EllipsoidDraw! *********" << endl;
	cerr << "   Drawing with quantity vector (";
	for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
		cerr << qtyVec[materialIndex];
		if (materialIndex < numberOfMaterials - 1) { cerr << ", "; }
		else { cerr << ")" << endl; }
	}

	cerr << "   Center point (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;
	cerr << "   Lengths " << lenVec[0] << ", " << lenVec[1] << ", " << lenVec[2] << endl;
	cerr << "   Angles " << angleVec[0] << ", " << angleVec[1] << ", " << angleVec[2] << endl;
	cerr << "   Drawing... ";

	angleVec[0] = -PI / 180.0*angleVec[0];
	angleVec[1] = -PI / 180.0*angleVec[1];
	angleVec[2] = -PI / 180.0*angleVec[2];

	Point3D pt(0, 0, 0);
	Point3D ptRel(0, 0, 0);
	Vector3 ptRelVec;
	for (int z = 0; z < fieldDim.z; ++z)
		for (int y = 0; y < fieldDim.y; ++y)
			for (int x = 0; x < fieldDim.x; ++x) {
				pt = Point3D(x, y, z);
				ptRel = pt - center;

				// rotate relative coordinate
				ptRelVec = Vector3((double)ptRel.x, (double)ptRel.y, (double)ptRel.z);
				ptRelVec.RotateZ(angleVec[2]);
				ptRelVec.RotateY(angleVec[1]);
				ptRelVec.RotateX(angleVec[0]);

				// test
				double insideTest = pow(ptRelVec.fX / lenVec[0], 2.0) + pow(ptRelVec.fY / lenVec[1], 2.0) + pow(ptRelVec.fZ / lenVec[2], 2.0);

				if (insideTest > 1.0) continue;

				setMediumNCMaterialQuantityVector(pt, qtyVec);

			}

	cerr << "complete." << endl;
	cerr << "**********************************" << endl;
}

void NCMaterialsPlugin::EllipseDraw(std::vector<float> _qtyVec, int _length, Point3D _center, double _angle, double _eccentricity) {

	#define PI 3.14159265

	std::vector<float> qtyVec = checkQuantities(_qtyVec);

	Point3D center = _center;
	int length = _length;
	double angle = _angle;
	double eccentricity = _eccentricity;

	Point3D pt(0, 0, 0);
	Point3D ptRel(0, 0, 0);

	cerr << "********** EllipseDraw! **********" << endl;
	cerr << "   Drawing with quantity vector (";
	for (int materialIndex = 0; materialIndex < numberOfMaterials; ++materialIndex) {
		cerr << qtyVec[materialIndex];
		if (materialIndex < numberOfMaterials - 1) { cerr << ", "; }
		else { cerr << ")" << endl; }
	}

	cerr << "   Center point (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;
	cerr << "   Semimajor axis length " << length << endl;
	cerr << "   Angle " << angle << endl;
	cerr << "   Eccentricity " << eccentricity << endl;
	cerr << "   Drawing... ";

	angle = PI * angle / 180.0;
	double ecc2 = pow(eccentricity, 2.0);
	double length2 = pow((double)length, 2.0);
	
	for (int y = 0; y < fieldDim.y; ++y)
		for (int x = 0; x < fieldDim.x; ++x) {
			pt = Point3D(x, y, 0);
			ptRel = pt - center;

			double anglePt = atan2(ptRel.y, ptRel.x);
			double anglePtRel = anglePt - angle;
			double ptDistance2 = (double)dotProduct(ptRel, ptRel);
			double boundaryDistance2 = length2 / (1-ecc2*pow(cos(anglePtRel), 2.0));
			
			if (ptDistance2 > boundaryDistance2) { continue; }

			for (int z = 0; z < fieldDim.z; ++z) {
				pt.z = z;
				setMediumNCMaterialQuantityVector(pt, qtyVec);
			}
		}

	cerr << "complete." << endl;
	cerr << "**********************************" << endl;
}

int NCMaterialsPlugin::dotProduct(Point3D _pt1, Point3D _pt2) {
	return _pt1.x*_pt2.x + _pt1.y*_pt2.y + _pt1.z*_pt2.z;
}

void NCMaterialsPlugin::overrideInitialization() {
    NCMaterialsInitialized = true;
    cerr << "NCMaterialsInitialized=" << NCMaterialsInitialized << endl;
}

std::string NCMaterialsPlugin::toString() {
    return "NCMaterials";
}


std::string NCMaterialsPlugin::steerableName() {
    return toString();
}


