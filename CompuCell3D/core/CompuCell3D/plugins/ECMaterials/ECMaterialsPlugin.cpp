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

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include "ECMaterialsPlugin.h"
#include "PublicUtilities/Vector3.h"


ECMaterialsPlugin::ECMaterialsPlugin() :
    pUtils(0),
    lockPtr(0),
    xmlData(0),
    numberOfMaterials(0),
    weightDistance(false),
    ECMaterialsInitialized(false)
{}

ECMaterialsPlugin::~ECMaterialsPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void ECMaterialsPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
	automaton = potts->getAutomaton();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    cerr << "Registering ECMaterials cell attributes..." << endl;

    potts->getCellFactoryGroupPtr()->registerClass(&ECMaterialCellDataAccessor);

	cerr << "Registering ECMaterials plugin..." << endl;

    potts->registerEnergyFunctionWithName(this, "ECMaterials");
    potts->registerCellGChangeWatcher(this);
    simulator->registerSteerableObject(this);

    if (_xmlData->getFirstElement("WeightEnergyByDistance")) {
        weightDistance = true;
    }

	// Boundary strategy for adhesion; based on implementation in AdhesionFlex
	boundaryStrategyAdh = BoundaryStrategy::getInstance();
	maxNeighborIndexAdh = 0;

	if (_xmlData->getFirstElement("Depth")) {
		maxNeighborIndexAdh = boundaryStrategyAdh->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
	}
	else {
		if (_xmlData->getFirstElement("NeighborOrder")) {
			maxNeighborIndexAdh = boundaryStrategyAdh->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());
		}
		else {
			maxNeighborIndexAdh = boundaryStrategyAdh->getMaxNeighborIndexFromNeighborOrder(1);
		}
	}

    // Gather XML user specifications
    CC3DXMLElementList ECMaterialNameXMLVec = _xmlData->getElements("ECMaterial");
    ECMaterialAdhesionXMLVec = _xmlData->getElements("ECAdhesion");
    CC3DXMLElementList ECMaterialAdvectionBoolXMLVec = _xmlData->getElements("ECMaterialAdvects");
    CC3DXMLElementList ECMaterialDurabilityXMLVec = _xmlData->getElements("ECMaterialDurability");
    ECMaterialRemodelingQuantityXMLVec = _xmlData->getElements("RemodelingQuantity");

    // generate name->integer index map for EC materials
    // generate array of pointers to EC materials according to user specification
    // assign material name from user specification
    set<std::string> ECMaterialNameSet;
    ECMaterialsVec.clear();
    ECMaterialNameIndexMap.clear();

    std::string ECMaterialName;

	cerr << "Declaring ECMaterials... " << endl;

    for (int i = 0; i < ECMaterialNameXMLVec.size(); ++i) {
        ECMaterialName = ECMaterialNameXMLVec[i]->getAttribute("Material");

		cerr << "   ECMaterial " << i << ": " << ECMaterialName << endl;

        if (!ECMaterialNameSet.insert(ECMaterialName).second) {
            ASSERT_OR_THROW(string("Duplicate ECMaterial Name=") + ECMaterialName + " specified in ECMaterials section ", false);
            continue;
        }

        ECMaterialsVec.push_back(ECMaterialComponentData());
        ECMaterialsVec[i].setName(ECMaterialName);

        ECMaterialNameIndexMap.insert(make_pair(ECMaterialName, i));
    }

    numberOfMaterials = ECMaterialsVec.size();

	cerr << "Number of ECMaterials defined: " << numberOfMaterials << endl;

    int ECMaterialIdx;
	bool ECMaterialIsAdvecting;

    // Assign optional advection specifications

	cerr << "Checking material advection options..." << endl;

    if (ECMaterialAdvectionBoolXMLVec.size() > 0){
        for (int i = 0; i < ECMaterialAdvectionBoolXMLVec.size(); i++) {
			ECMaterialName = ECMaterialAdvectionBoolXMLVec[i]->getAttribute("Material");
			ECMaterialIsAdvecting = ECMaterialAdvectionBoolXMLVec[i]->getBool();

			cerr << "   ECMaterial " << ECMaterialName << " advection mode: " << ECMaterialIsAdvecting << endl;

            ECMaterialIdx = getECMaterialIndexByName(ECMaterialName);
            ECMaterialsVec[ECMaterialIdx].setTransferable(ECMaterialIsAdvecting);
        }
    }

    // Assign barrier Lagrange multiplier to each material from user specification

	float durabilityLM;

	cerr << "Assigning ECMaterial durability coefficients..." << endl;

    if (ECMaterialDurabilityXMLVec.size() > 0) {
        for (int i = 0; i < ECMaterialDurabilityXMLVec.size(); ++i) {
			ECMaterialName = ECMaterialDurabilityXMLVec[i]->getAttribute("Material");
			durabilityLM = (float)ECMaterialDurabilityXMLVec[i]->getDouble();

			cerr << "   ECMaterial " << ECMaterialName << " barrier Lagrange multiplier: " << durabilityLM << endl;
			
			ECMaterialIdx = getECMaterialIndexByName(ECMaterialName);
            ECMaterialsVec[ECMaterialIdx].setDurabilityLM(durabilityLM);
        }
    }

    // Initialize quantity vector field

	cerr << "Initializing ECMaterials quantity field..." << endl;

    fieldDim=potts->getCellFieldG()->getDim();
    //      Unsure if constructor can be used as initial value
    // ECMaterialsField = new WatchableField3D<ECMaterialsData *>(fieldDim, ECMaterialsData());
    //      Trying 0, as in Potts3D::createCellField
    ECMaterialsField = new WatchableField3D<ECMaterialsData *>(fieldDim, 0);

    // initialize EC material quantity vector field values
    // default is all first ECM component

	cerr << "Initializing ECMaterials quantity field values..." << endl;

    Point3D pt(0,0,0);
    ECMaterialsData *ECMaterialsDataLocal;
    for (int z = 0; z < fieldDim.z; ++z){
        for (int y = 0; y < fieldDim.y; ++y){
            for (int x = 0; x < fieldDim.x; ++x){
				pt = Point3D(x, y, z);

				ECMaterialsDataLocal = new ECMaterialsData();
				std::vector<float> & ECMaterialsQuantityVec = ECMaterialsDataLocal->ECMaterialsQuantityVec;
				std::vector<float> ECMaterialsQuantityVecNew(numberOfMaterials);
				ECMaterialsQuantityVecNew[0] = 1.0;
				ECMaterialsQuantityVec = ECMaterialsQuantityVecNew;
                ECMaterialsField->set(pt, ECMaterialsDataLocal);
            }
        }
    }

	// Field design, if specified

	if (_xmlData->findElement("FieldDesign")) {

		cerr << "Initializing field designs" << endl;

		CC3DXMLElementList ECMaterialFieldDesignVec = _xmlData->getElements("FieldDesign");
		
		// in case user specified multiple FieldDesign elements, loop over all discovered

		bool designIsValid;

		for (int fieldDesignIndex = 0; fieldDesignIndex < ECMaterialFieldDesignVec.size(); ++fieldDesignIndex) {

			CC3DXMLElementList FieldDesigns = ECMaterialFieldDesignVec[fieldDesignIndex]->getElements("Design");

			// apply design

			for (CC3DXMLElementList::iterator designItr = FieldDesigns.begin(); designItr != FieldDesigns.end(); ++designItr) {

				// get specified materials

				CC3DXMLElementList theseMaterials = (*designItr)->getElements("ECMaterial");
				std::vector<float> thisQuantityVector(numberOfMaterials, 0.0);

				bool aMaterialWasSpecified = false;

				for (CC3DXMLElementList::iterator materialItr = theseMaterials.begin(); materialItr != theseMaterials.end(); ++materialItr) {
					int materialIndex = getECMaterialIndexByName((*materialItr)->getAttribute("Material"));
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
	
}

void ECMaterialsPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);

}

void ECMaterialsPlugin::handleEvent(CC3DEvent & _event) {
    if (_event.id == CHANGE_NUMBER_OF_WORK_NODES) {

        //vectorized variables for convenient parallel access
        // unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();

    }
}


double ECMaterialsPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    //cerr<<"ChangeEnergy"<<endl;
    if (!ECMaterialsInitialized) {
        pUtils->setLock(lockPtr);
        initializeECMaterials();
        pUtils->unsetLock(lockPtr);
    }

    double energy = 0;
    double distance = 0;
    Neighbor neighbor;
    vector<float> targetQuantityVec(numberOfMaterials);

    // Target medium and durability
    if (oldCell == 0) {
        targetQuantityVec = ECMaterialsField->get(pt)->ECMaterialsQuantityVec;
        energy += ECMaterialDurabilityEnergy(targetQuantityVec);
    }
    vector<float> copyQuantityVector(numberOfMaterials);
    if ((newCell == 0) && (oldCell != 0)) {
        copyQuantityVector = calculateCopyQuantityVec(oldCell, pt);
    }

    CellG *nCell = 0;
    WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

    if (weightDistance) {
        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexAdh; ++nIdx) {
            neighbor = boundaryStrategyAdh->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);

            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            distance = neighbor.distance;

            nCell = fieldG->get(neighbor.pt);
			vector<float> nQuantityVec = ECMaterialsField->get(neighbor.pt)->ECMaterialsQuantityVec;

            if (nCell != oldCell) {
                if (nCell == 0) {
                    energy -= ECMaterialContactEnergy(oldCell, nQuantityVec) / neighbor.distance;
                }
                else if (oldCell == 0) {
                    energy -= ECMaterialContactEnergy(nCell, targetQuantityVec) / neighbor.distance;
                }
            }
            if (nCell != newCell) {
                if (nCell == 0) {
                    energy += ECMaterialContactEnergy(newCell, nQuantityVec) / neighbor.distance;
                }
                else if (newCell == 0) {
                    energy += ECMaterialContactEnergy(nCell, copyQuantityVector) / neighbor.distance;
                }
            }
        }
    }
    else {
        //default behaviour  no energy weighting

        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexAdh; ++nIdx) {
            neighbor = boundaryStrategyAdh->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nCell = fieldG->get(neighbor.pt);
			vector<float> nQuantityVec = ECMaterialsField->get(neighbor.pt)->ECMaterialsQuantityVec;

            if (nCell != oldCell) {
                if (nCell == 0) {
                    energy -= ECMaterialContactEnergy(oldCell, nQuantityVec);
                }
                else if (oldCell == 0) {
                    energy -= ECMaterialContactEnergy(nCell, targetQuantityVec);
                }
            }
            if (nCell != newCell) {
                if (nCell == 0) {
                    energy += ECMaterialContactEnergy(newCell, nQuantityVec);
                }
                else if (newCell == 0) {
                    energy += ECMaterialContactEnergy(nCell, copyQuantityVector);
                }

            }
        }
    }

    return energy;
}

double ECMaterialsPlugin::ECMaterialContactEnergy(const CellG *cell, std::vector<float> _qtyVec) {

    double energy = 0.0;

    if (cell != 0) {
		std::vector<float> AdhesionCoefficients = ECMaterialCellDataAccessor.get(cell->extraAttribPtr)->AdhesionCoefficients;
        for (int i = 0; i < AdhesionCoefficients.size() ; ++i){
            energy += AdhesionCoefficients[i]*_qtyVec[i];
        }
    }

    return energy;

}

double ECMaterialsPlugin::ECMaterialDurabilityEnergy(std::vector<float> _qtyVec) {

    double energy = 0.0;
    float thisEnergy;

    for (unsigned int i = 0; i < _qtyVec.size(); ++i) {
        thisEnergy = _qtyVec[i]* ECMaterialsVec[i].getDurabilityLM();
        if (thisEnergy > 0.0) {energy += (double) thisEnergy;}
    }

    return energy;

}

void ECMaterialsPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    // If source agent is a cell and target agent is the medium, then target EC materials are removed
    // If source agent is the medium, materials advect

	ECMaterialsData *ECMaterialsDataLocal = ECMaterialsField->get(pt);
	std::vector<float> & ECMaterialsQuantityVec = ECMaterialsDataLocal->ECMaterialsQuantityVec;
	std::vector<float> ECMaterialsQuantityVecNew(numberOfMaterials);

    if (newCell) { // Source agent is a cell
        if (!oldCell){
			ECMaterialsQuantityVec = ECMaterialsQuantityVecNew;
			ECMaterialsField->set(pt, ECMaterialsDataLocal);
        }
    }
    else { // Source agent is a medium
        if (oldCell){
			ECMaterialsQuantityVec = calculateCopyQuantityVec(oldCell, pt);
			ECMaterialsField->set(pt, ECMaterialsDataLocal);
        }
    }

}

std::vector<float> ECMaterialsPlugin::calculateCopyQuantityVec(const CellG * _cell, const Point3D &pt) {

    std::vector<float> copyQuantityVec(numberOfMaterials);

    // Calculate copy quantity vector
    // quantity vector is mean of all transferable neighborhood components + target cell remodeling quantity
    CellG *neighborCell;
	if (_cell) { copyQuantityVec = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity; }

    float numberOfMediumNeighbors = 1.0;
    std::vector<float> neighborQuantityVector(numberOfMaterials);
    std::vector<Neighbor> neighbors = getFirstOrderNeighbors(pt);
    WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    for (int nIdx = 0; nIdx < neighbors.size(); ++nIdx){
        neighborCell = fieldG->get(neighbors[nIdx].pt);
        if (neighborCell) {continue;}
		neighborQuantityVector = ECMaterialsField->get(neighbors[nIdx].pt)->ECMaterialsQuantityVec;
		for (int i = 0; i < ECMaterialsVec.size(); ++i) {
			if ( !(ECMaterialsVec[i].getTransferable()) ) { neighborQuantityVector[i] = 0.0; }
		}

        for (int i = 0; i < copyQuantityVec.size(); ++i) {copyQuantityVec[i] += neighborQuantityVector[i];}
        numberOfMediumNeighbors += 1.0;
    }

    for (int i = 0; i < copyQuantityVec.size(); ++i){copyQuantityVec[i] /= numberOfMediumNeighbors;}

    std::vector<float> copyQuantityVecChecked = ECMaterialsPlugin::checkQuantities(copyQuantityVec);

	return copyQuantityVecChecked;

}

void ECMaterialsPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
        set<unsigned char> cellTypesSet;

}


void ECMaterialsPlugin::initializeECMaterials() {
    
	cerr << "ECMaterials plugin initialization begin: " << ECMaterialsInitialized << endl;

	if (ECMaterialsInitialized)//we double-check this flag to makes sure this function does not get called multiple times by different threads
        return;


	// initialize cell type->remodeling quantity map
	//      first collect inputs as a 2D array by type and material ids, then assemble map

	cerr << "Initializing remodeling quantities..." << endl;

	int thisTypeId;
	int maxTypeId = (int)automaton->getMaxTypeId();
	int ECMaterialIdx;
	double thisRemodelingQuantity;
	std::string ECMaterialName;
	std::string cellTypeName;
	std::vector<std::string> cellTypeNamesByTypeId;
	std::vector<std::vector<float> > RemodelingQuantityByTypeId(maxTypeId + 1, vector<float>(numberOfMaterials));
	typeToRemodelingQuantityMap.clear();

	for (int i = 0; i < ECMaterialRemodelingQuantityXMLVec.size(); ++i) {
		cellTypeName = ECMaterialRemodelingQuantityXMLVec[i]->getAttribute("CellType");
		ECMaterialName = ECMaterialRemodelingQuantityXMLVec[i]->getAttribute("Material");
		thisRemodelingQuantity = ECMaterialRemodelingQuantityXMLVec[i]->getDouble();

		cerr << "   Initializing (" << cellTypeName << ", " << ECMaterialName << " ): " << thisRemodelingQuantity << endl;

		thisTypeId = (int)automaton->getTypeId(cellTypeName);
		ECMaterialIdx = getECMaterialIndexByName(ECMaterialName);
		cellTypeNamesByTypeId.push_back(cellTypeName);
		RemodelingQuantityByTypeId[thisTypeId][ECMaterialIdx] = (float)thisRemodelingQuantity;
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

	cerr << "Initializing cell-ECMaterial interface adhesion coefficients by cell type and material component..." << endl;

	AdhesionCoefficientsByTypeId.clear();
	std::vector<std::vector<float> > AdhesionCoefficientsByTypeId(maxTypeId + 1, std::vector<float>(numberOfMaterials));
	double thisAdhesionCoefficient;
	for (int i = 0; i < ECMaterialAdhesionXMLVec.size(); ++i) {
		cellTypeName = ECMaterialAdhesionXMLVec[i]->getAttribute("CellType");
		ECMaterialName = ECMaterialAdhesionXMLVec[i]->getAttribute("Material");
		thisAdhesionCoefficient = ECMaterialAdhesionXMLVec[i]->getDouble();

		cerr << "   Initializing (" << cellTypeName << ", " << ECMaterialName << " ): " << thisAdhesionCoefficient << endl;

		thisTypeId = (int)automaton->getTypeId(cellTypeName);
		ECMaterialIdx = getECMaterialIndexByName(ECMaterialName);
		AdhesionCoefficientsByTypeId[thisTypeId][ECMaterialIdx] = (float)thisAdhesionCoefficient;
	}

	// assign remodeling quantities and adhesion coefficients to cells by type and material name from user specification

	cerr << "Assigning remodeling quantities and adhesion coefficients to cells..." << endl;

	std::map<std::string, std::vector<float> >::iterator mitr;
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	CellInventory * cellInventoryPtr = &potts->getCellInventory();
	for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
		cell = cellInventoryPtr->getCell(cInvItr);
		std::vector<float> & RemodelingQuantity = ECMaterialCellDataAccessor.get(cell->extraAttribPtr)->RemodelingQuantity;
		std::vector<float> & AdhesionCoefficients = ECMaterialCellDataAccessor.get(cell->extraAttribPtr)->AdhesionCoefficients;

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

    ECMaterialsInitialized = true;

	cerr << "ECMaterials plugin initialization complete: " << ECMaterialsInitialized << endl;

}

void ECMaterialsPlugin::setMaterialNameVector() {
	fieldNameVec.clear();
	std::map<std::string, int>::iterator mitr;
	for (mitr = ECMaterialNameIndexMap.begin(); mitr != ECMaterialNameIndexMap.end(); ++mitr) {
		fieldNameVec.push_back(mitr->first);
	}
}

std::vector<float> ECMaterialsPlugin::checkQuantities(std::vector<float> _qtyVec) {
	float qtySum = 0.0;
	for (int i = 0; i < _qtyVec.size(); ++i) {
		if (_qtyVec[i] < 0.0 || isnan(_qtyVec[i])) { _qtyVec[i] = 0.0; }
		else if (_qtyVec[i] > 1.0) { _qtyVec[i] = 1.0; }
		qtySum += _qtyVec[i];
    }
	if (qtySum > 1.0) { for (int i = 0; i < _qtyVec.size(); ++i) { _qtyVec[i] /= qtySum; } }
    return _qtyVec;
}

void ECMaterialsPlugin::setRemodelingQuantityByName(const CellG * _cell, std::string _ECMaterialName, float _quantity) {
    int _idx = getECMaterialIndexByName(_ECMaterialName);
    setRemodelingQuantityByIndex(_cell, _idx, _quantity);
}

void ECMaterialsPlugin::setRemodelingQuantityByIndex(const CellG * _cell, int _idx, float _quantity) {
	std::vector<float> & RemodelingQuantity = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	if (_idx >= 0 && _idx < RemodelingQuantity.size()) { RemodelingQuantity[_idx] = _quantity; }
}

void ECMaterialsPlugin::setRemodelingQuantityVector(const CellG * _cell, std::vector<float> _quantityVec) {
	std::vector<float> & RemodelingQuantity = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	RemodelingQuantity = _quantityVec;
}

void ECMaterialsPlugin::assignNewRemodelingQuantityVector(const CellG * _cell, unsigned int _numMtls) {
	std::vector<float> RemodelingQuantityNew(_numMtls);
	std::vector<float> & RemodelingQuantity = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
	RemodelingQuantity = RemodelingQuantityNew;
}

void ECMaterialsPlugin::setMediumECMaterialQuantityByName(const Point3D &pt, std::string _ECMaterialName, float _quantity) {
    setMediumECMaterialQuantityByIndex(pt, getECMaterialIndexByName(_ECMaterialName), _quantity);
}

void ECMaterialsPlugin::setMediumECMaterialQuantityByIndex(const Point3D &pt, int _idx, float _quantity) {
    ECMaterialsData *ECMaterialsDataLocal = ECMaterialsField->get(pt);
	std::vector<float> & ECMaterialsQuantityVec = ECMaterialsDataLocal->ECMaterialsQuantityVec;
	ECMaterialsQuantityVec[_idx] = _quantity;
    ECMaterialsField->set(pt, ECMaterialsDataLocal);
}

void ECMaterialsPlugin::setMediumECMaterialQuantityVector(const Point3D &pt, std::vector<float> _quantityVec) {
    ECMaterialsData *ECMaterialsDataLocal = ECMaterialsField->get(pt);
	std::vector<float> & ECMaterialsQuantityVec = ECMaterialsDataLocal->ECMaterialsQuantityVec;
	ECMaterialsQuantityVec = _quantityVec;
    ECMaterialsField->set(pt, ECMaterialsDataLocal);
}

void ECMaterialsPlugin::assignNewMediumECMaterialQuantityVector(const Point3D &pt, unsigned int _numMtls) {
    ECMaterialsData *ECMaterialsDataLocal = ECMaterialsField->get(pt);
	std::vector<float> ECMaterialsQuantityVecNew(_numMtls);
	std::vector<float> & ECMaterialsQuantityVec = ECMaterialsDataLocal->ECMaterialsQuantityVec;
	ECMaterialsQuantityVec = ECMaterialsQuantityVecNew;
	ECMaterialsField->set(pt, ECMaterialsDataLocal);
}

void ECMaterialsPlugin::setECMaterialDurabilityByName(std::string _ECMaterialName, float _durabilityLM) {
    setECMaterialDurabilityByIndex(getECMaterialIndexByName(_ECMaterialName), _durabilityLM);
}

void ECMaterialsPlugin::setECMaterialDurabilityByIndex(int _idx, float _durabilityLM) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx < ECMaterialsVec.size() ) {
        ECMaterialsVec[_idx].setDurabilityLM(_durabilityLM);
    }
}

void ECMaterialsPlugin::setECMaterialAdvectingByName(std::string _ECMaterialName, bool _isAdvecting) {
    setECMaterialAdvectingByIndex(getECMaterialIndexByName(_ECMaterialName), _isAdvecting);
}

void ECMaterialsPlugin::setECMaterialAdvectingByIndex(int _idx, bool _isAdvecting) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx < ECMaterialsVec.size() ){
        ECMaterialsVec[_idx].setTransferable(_isAdvecting);
    }
}

void ECMaterialsPlugin::setECAdhesionByCell(const CellG *_cell, std::vector<float> _adhVec) {
    vector<float> & AdhesionCoefficients = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->AdhesionCoefficients;
	AdhesionCoefficients = _adhVec;
}

float ECMaterialsPlugin::getRemodelingQuantityByName(const CellG * _cell, std::string _ECMaterialName) {
    return getRemodelingQuantityByIndex(_cell, getECMaterialIndexByName(_ECMaterialName));
}

float ECMaterialsPlugin::getRemodelingQuantityByIndex(const CellG * _cell, int _idx) {
	std::vector<float> remodelingQuantityVector = getRemodelingQuantityVector(_cell);
	if (_idx < 0 || _idx > remodelingQuantityVector.size()-1 ) {
        ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!" , false);
        return -1.0;
    }
	return remodelingQuantityVector[_idx];
}

std::vector<float> ECMaterialsPlugin::getRemodelingQuantityVector(const CellG * _cell) {
    std::vector<float> remodelingQuantityVector(numberOfMaterials);
    if (_cell) {
		std::vector<float> & remodelingQuantityVector = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->RemodelingQuantity;
    }
    return remodelingQuantityVector;
}

float ECMaterialsPlugin::getMediumECMaterialQuantityByName(const Point3D &pt, std::string _ECMaterialName) {
    return getMediumECMaterialQuantityByIndex(pt, getECMaterialIndexByName(_ECMaterialName));
}

float ECMaterialsPlugin::getMediumECMaterialQuantityByIndex(const Point3D &pt, int _idx) {
	std::vector<float> ECMaterialsQuantityVec = getMediumECMaterialQuantityVector(pt);
	if (_idx < 0 || _idx > ECMaterialsQuantityVec.size()-1) {
		ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!", false);
		return -1.0;
	}
	return ECMaterialsQuantityVec[_idx];
}

std::vector<float> ECMaterialsPlugin::getMediumECMaterialQuantityVector(const Point3D &pt) {
	vector<float> & ECMaterialsQuantityVec = ECMaterialsField->get(pt)->ECMaterialsQuantityVec;
    return ECMaterialsQuantityVec;
}

std::vector<float> ECMaterialsPlugin::getMediumAdvectingECMaterialQuantityVector(const Point3D &pt) {
    std::vector<float> _qtyVec = getMediumECMaterialQuantityVector(pt);
    std::vector<ECMaterialComponentData> _ecmVec = getECMaterialsVec();
    for (int i = 0; i < _qtyVec.size(); ++i) {
		if (!_ecmVec[i].getTransferable()) {
            _qtyVec[i] = 0.0;
        }
    }
    return _qtyVec;
}

float ECMaterialsPlugin::getECMaterialDurabilityByName(std::string _ECMaterialName) {
    return getECMaterialDurabilityByIndex(getECMaterialIndexByName(_ECMaterialName));
}

float ECMaterialsPlugin::getECMaterialDurabilityByIndex(int _idx) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx < ECMaterialsVec.size()){
        return ECMaterialsVec[_idx].getDurabilityLM();
    }
	ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!", false);
    return 0;
}

bool ECMaterialsPlugin::getECMaterialAdvectingByName(std::string _ECMaterialName) {
    return getECMaterialAdvectingByIndex(getECMaterialIndexByName(_ECMaterialName));
}

bool ECMaterialsPlugin::getECMaterialAdvectingByIndex(int _idx) {
    // if index exceeds number of materials, then ignore it
    if (_idx >= 0 && _idx > ECMaterialsVec.size()-1 ){
        return ECMaterialsVec[_idx].getTransferable();
    }
	ASSERT_OR_THROW(std::string("Material index ") + std::to_string(_idx) + " out of range!", false);
    return 0;
}

int ECMaterialsPlugin::getECMaterialIndexByName(std::string _ECMaterialName){
	std::map<std::string, int> ecmaterialNameIndexMap = getECMaterialNameIndexMap();
    std::map<std::string, int>::iterator mitr = ecmaterialNameIndexMap.find(_ECMaterialName);
    if ( mitr != ecmaterialNameIndexMap.end() ){
        return mitr->second;
    }
    return -1;
}

std::vector<float> ECMaterialsPlugin::getECAdhesionByCell(const CellG *_cell) {
    std::vector<float> & adhVec = ECMaterialCellDataAccessor.get(_cell->extraAttribPtr)->AdhesionCoefficients;
    return adhVec;
}

std::vector<float> ECMaterialsPlugin::getECAdhesionByCellTypeId(int _idx) {
    std::vector<float> _adhVec(numberOfMaterials);
    if (_idx < 0 || _idx > AdhesionCoefficientsByTypeId.size()-1) {ASSERT_OR_THROW("Material index out of range!" , false);}
    else {_adhVec = AdhesionCoefficientsByTypeId[_idx];}
    return _adhVec;
}

std::vector<Neighbor> ECMaterialsPlugin::getFirstOrderNeighbors(const Point3D &pt) {
    // initialize neighborhood according to Potts neighborhood
    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
    std::vector<Neighbor> neighbors;
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
		Neighbor neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt), nIdx);
		if (!neighbor.distance) {
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
        neighbors.push_back(neighbor);
    }
    return neighbors;
}

void ECMaterialsPlugin::ParaDraw(std::vector<float> _qtyVec, Point3D _startPos, Point3D _lenVec1, Point3D _lenVec2, Point3D _lenVec3) {

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
	for (int z = 0; z < fieldDim.z; ++z) {
		for (int y = 0; y < fieldDim.y; ++y) {
			for (int x = 0; x < fieldDim.x; ++x) {
				pt = Point3D(x, y, z);
				ptRel = pt - StartPos;
				
				int dotProductPt1 = dotProduct(ptRel, lenVec1);
				if ( (dotProductPt1 < 0) || (dotProductPt1 > len12) ) continue;
				int dotProductPt2 = dotProduct(ptRel, lenVec2);
				if ( (dotProductPt2 < 0) || (dotProductPt2 > len22) ) continue;
				int dotProductPt3 = dotProduct(ptRel, lenVec3);
				if ( (dotProductPt3 < 0) || (dotProductPt3 > len32) ) continue;

				setMediumECMaterialQuantityVector(pt, qtyVec);

			}
		}
	}

	cerr << "complete." << endl;
	cerr << "*********************************" << endl;

}

void ECMaterialsPlugin::CylinderDraw(std::vector<float> _qtyVec, short _radius, Point3D _startPos, Point3D _lenVec) {

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
	for (int z = 0; z < fieldDim.z; ++z) {
		for (int y = 0; y < fieldDim.y; ++y) {
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
				
				setMediumECMaterialQuantityVector(pt, qtyVec);

			}
		}
	}

	cerr << "complete." << endl;
	cerr << "*********************************" << endl;

}

void ECMaterialsPlugin::EllipsoidDraw(std::vector<float> _qtyVec, Point3D _center, std::vector<short> _lenVec, std::vector<double> _angleVec) {

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
	for (int z = 0; z < fieldDim.z; ++z) {
		for (int y = 0; y < fieldDim.y; ++y) {
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

				setMediumECMaterialQuantityVector(pt, qtyVec);

			}
		}
	}

	cerr << "complete." << endl;
	cerr << "**********************************" << endl;
}

void ECMaterialsPlugin::EllipseDraw(std::vector<float> _qtyVec, int _length, Point3D _center, double _angle, double _eccentricity) {

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
	
	for (int y = 0; y < fieldDim.y; ++y) {
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
				setMediumECMaterialQuantityVector(pt, qtyVec);
			}
		}
	}

	cerr << "complete." << endl;
	cerr << "**********************************" << endl;
}

int ECMaterialsPlugin::dotProduct(Point3D _pt1, Point3D _pt2) {
	return _pt1.x*_pt2.x + _pt1.y*_pt2.y + _pt1.z*_pt2.z;
}

void ECMaterialsPlugin::overrideInitialization() {
    ECMaterialsInitialized = true;
    cerr << "ECMaterialsInitialized=" << ECMaterialsInitialized << endl;
}

std::string ECMaterialsPlugin::toString() {
    return "ECMaterials";
}


std::string ECMaterialsPlugin::steerableName() {
    return toString();
}


