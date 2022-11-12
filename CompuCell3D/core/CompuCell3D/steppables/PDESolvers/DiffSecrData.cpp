

#include <PublicUtilities/StringUtils.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/CC3DExceptions.h>
#include <XMLUtils/CC3DXMLElement.h>

#include "DiffSecrData.h"
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;
using namespace std;

std::string DiffusionData::steerableName() {
    return "DiffusionData";
}

void DiffusionData::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {


    if (_xmlData->findElement("DiffusionConstant"))
        diffConst = _xmlData->getFirstElement("DiffusionConstant")->getDouble();

    if (_xmlData->findElement("GlobalDiffusionConstant"))
        diffConst = _xmlData->getFirstElement("GlobalDiffusionConstant")->getDouble();


    if (_xmlData->findElement("DecayConstant"))
        decayConst = _xmlData->getFirstElement("DecayConstant")->getDouble();

    if (_xmlData->findElement("GlobalDecayConstant"))
        decayConst = _xmlData->getFirstElement("GlobalDecayConstant")->getDouble();


    //cell-type dependent coeficients
    if (_xmlData->findElement("DiffusionCoefficient")) {
        CC3DXMLElementList diffCoefXMLVec = _xmlData->getElements("DiffusionCoefficient");
        for (unsigned int i = 0; i < diffCoefXMLVec.size(); ++i) {

			diffCoefTypeNameMap.insert(make_pair(diffCoefXMLVec[i]->getAttribute("CellType"),diffCoefXMLVec[i]->getDouble()));
			
        }
	}
	

    if (_xmlData->findElement("DecayCoefficient")) {
        CC3DXMLElementList decayCoefXMLVec = _xmlData->getElements("DecayCoefficient");
        for (unsigned int i = 0; i < decayCoefXMLVec.size(); ++i) {

            decayCoefTypeNameMap.insert(make_pair(decayCoefXMLVec[i]->getAttribute("CellType"), decayCoefXMLVec[i]->getDouble()));

        }
    }


    if (_xmlData->findElement("DeltaX"))
        deltaX = _xmlData->getFirstElement("DeltaX")->getDouble();

    if (_xmlData->findElement("DeltaT"))
        deltaT = _xmlData->getFirstElement("DeltaT")->getDouble();

    if (_xmlData->findElement("ExtraTimesPerMCS"))
        extraTimesPerMCS = _xmlData->getFirstElement("ExtraTimesPerMCS")->getUInt();


    if (_xmlData->findElement("UseBoxWatcher"))
        useBoxWatcher = true;
    if (_xmlData->findElement("DoNotDiffuseTo")) {
        CC3DXMLElementList doNotDiffuseToXMLVec = _xmlData->getElements("DoNotDiffuseTo");
        for (unsigned int i = 0; i < doNotDiffuseToXMLVec.size(); ++i) {
            avoidTypeNameSet.insert(doNotDiffuseToXMLVec[i]->getText());
        }
    }

    if (_xmlData->findElement("DoNotDecayIn")) {
        CC3DXMLElementList doNotDecayInXMLVec = _xmlData->getElements("DoNotDecayIn");
        for (unsigned int i = 0; i < doNotDecayInXMLVec.size(); ++i) {
            avoidDecayInTypeNameSet.insert(doNotDecayInXMLVec[i]->getText());
        }
    }

    if (_xmlData->findElement("CouplingTerm")) {
        CC3DXMLElementList couplingTermXMLVec = _xmlData->getElements("CouplingTerm");
        for (unsigned int i = 0; i < couplingTermXMLVec.size(); ++i) {

            CouplingData couplData;
            if (couplingTermXMLVec[i]->findAttribute("InteractingFieldName")) {
                couplData.intrFieldName = couplingTermXMLVec[i]->getAttribute("InteractingFieldName");
            }

            if (couplingTermXMLVec[i]->findAttribute("CouplingCoefficient")) {
                couplData.couplingCoef = couplingTermXMLVec[i]->getAttributeAsDouble("CouplingCoefficient");
            }

            couplingDataVec.push_back(couplData);
        }
    }

    CC3DXMLElement *thresholdsXMLElement = _xmlData->getFirstElement("ConcentrationThresholds");
    if (thresholdsXMLElement) {
        useThresholds = true;
        if (thresholdsXMLElement->findAttribute("MinConcentration")) {
            double tempMinConcentration = thresholdsXMLElement->getAttributeAsDouble("MinConcentration");
            if (tempMinConcentration <= minConcentration)
                throw CC3DException("MinConCentration is smaller than minimum allowed value");
            minConcentration = tempMinConcentration;
        }

        if (thresholdsXMLElement->findAttribute("MaxConcentration")) {
            double tempMaxConcentration = thresholdsXMLElement->getAttributeAsDouble("MaxConcentration");
            if (tempMaxConcentration >= maxConcentration)
                throw CC3DException("MaxConCentration is bigger than maximum allowed vaule");
            maxConcentration = tempMaxConcentration;
        }

        if (maxConcentration <= minConcentration)
            throw CC3DException("MaxConcentration has to be greater than MinCoCentration");
    }

    if (_xmlData->findElement("ConcentrationFileName"))
        concentrationFileName = _xmlData->getFirstElement("ConcentrationFileName")->getText();

    if (_xmlData->findElement("InitialConcentrationExpression"))
        initialConcentrationExpression = _xmlData->getFirstElement("InitialConcentrationExpression")->getText();

    // notice that field name may be extracted from this element  <DiffusionField Name="FGF">
    if (_xmlData->findElement("FieldName") && !fieldName.size())
        fieldName = _xmlData->getFirstElement("FieldName")->getText();

    if (_xmlData->findElement("AdditionalTerm"))
        additionalTerm = _xmlData->getFirstElement("AdditionalTerm")->getText();

	if(_xmlData->findElement("CallUserFuncs"))
        userFuncFlag=_xmlData->getFirstElement("CallUserFuncs")->getUInt();
        
    if(_xmlData->findElement("CFunc"))
        userFuncFlag=1;
    
    if(_xmlData->findElement("FuncName"))
        funcName=_xmlData->getFirstElement("FuncName")->getText();
    
    if(_xmlData->findElement("FieldDependencies"))
        FieldDependenciesSTR=_xmlData->getFirstElement("FieldDependencies")->cdata;
    
    parseStringIntoList(FieldDependenciesSTR , fieldDependencies, ",");

    for(int i = 0; i< fieldDependencies.size(); i++) {
        CC3D_Log(LOG_DEBUG) << "fieldDependencies: " << fieldDependencies[i];
    }
    CC3D_Log(LOG_DEBUG) << *this;

}


bool DiffusionData::getVariableDiffusionCoeeficientFlag() { return variableDiffusionCoefficientFlag; }

void DiffusionData::DeltaX(float _dx) { deltaX = _dx; }

void DiffusionData::DeltaT(float _dt) { deltaT = _dt; }

void DiffusionData::UseBoxWatcher(bool _flag) { useBoxWatcher = _flag; }

void DiffusionData::CouplingTerm(std::string _interactingFieldName, float _couplingCoefficient) {
    CouplingData couplData;
    couplData.intrFieldName = _interactingFieldName;
    couplData.couplingCoef = _couplingCoefficient;
    couplingDataVec.push_back(couplData);

}

void DiffusionData::MinConcentrationThreshold(float _min) {
    useThresholds = true;
    minConcentration = _min;

}

void DiffusionData::MaxConcentrationThreshold(float _max) {
    useThresholds = true;
    maxConcentration = _max;
}


void DiffusionData::FieldName(std::string _fieldName) {
    fieldName = _fieldName;
}

void DiffusionData::ConcentrationFileName(std::string _concFieldName) {
    concentrationFileName = _concFieldName;
}

void DiffusionData::DiffusionConstant(float _diffConst) { diffConst = _diffConst; }

void DiffusionData::DecayConstant(float _decayConst) { decayConst = _decayConst; }


void DiffusionData::DoNotDiffuseTo(std::string _typeName) {
    avoidTypeNameSet.insert(_typeName);
}

void DiffusionData::DoNotDecayIn(std::string _typeName) {
    avoidDecayInTypeNameSet.insert(_typeName);
}


void DiffusionData::initialize(Automaton *_automaton) {

    avoidDecayInIdSet.clear();

    for (std::set<std::string>::iterator sitr = avoidDecayInTypeNameSet.begin();
         sitr != avoidDecayInTypeNameSet.end(); ++sitr) {
        unsigned char typeId = _automaton->getTypeId(*sitr);
        avoidDecayInIdSet.insert(typeId);
    }

    avoidTypeIdSet.clear();
    for (std::set<std::string>::iterator sitr = avoidTypeNameSet.begin(); sitr != avoidTypeNameSet.end(); ++sitr) {
        unsigned char typeId = _automaton->getTypeId(*sitr);
        avoidTypeIdSet.insert(typeId);
    }

    //clearing up decayCoef and diffCoef - we actually set them to correspond to the global diffusionConst and decayConstanmt
    for (int i = 0; i < UCHAR_MAX + 1; ++i) {
        //decayCoef[i]=0.0;
        decayCoef[i] = decayConst;
    }

    for (int i = 0; i < UCHAR_MAX + 1; ++i) {
        diffCoef[i] = diffConst;
    }


    //here we use by-type definitions - if any are present
    for (std::map<std::string, float>::iterator mitr = decayCoefTypeNameMap.begin();
         mitr != decayCoefTypeNameMap.end(); ++mitr) {
        unsigned char typeId = _automaton->getTypeId(mitr->first);
        decayCoef[typeId] = mitr->second;
    }

    for (std::map<std::string, float>::iterator mitr = diffCoefTypeNameMap.begin();
         mitr != diffCoefTypeNameMap.end(); ++mitr) {
        unsigned char typeId = _automaton->getTypeId(mitr->first);
        diffCoef[typeId] = mitr->second;
        variableDiffusionCoefficientFlag = true;
    }


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string SecretionData::steerableName() {
    return "SecretionData";
}


void SecretionData::Secretion(std::string _typeName, float _secretionConst) {
    typeNameSecrConstMap.insert(make_pair(_typeName, _secretionConst));
    secrTypesNameSet.insert("Secretion");
}

void SecretionData::SecretionOnContact(std::string _secretingTypeName, std::string _onContactWithTypeName,
                                       float _secretionConst) {
    std::map<std::string, SecretionOnContactData>::iterator mitr;
    mitr = typeNameSecrOnContactDataMap.find(_secretingTypeName);

    if (mitr != typeNameSecrOnContactDataMap.end()) {
        mitr->second.contactCellMapTypeNames.insert(make_pair(_onContactWithTypeName, _secretionConst));
    } else {
        SecretionOnContactData secrOnContactData;
        secrOnContactData.contactCellMapTypeNames.insert(make_pair(_onContactWithTypeName, _secretionConst));
        typeNameSecrOnContactDataMap.insert(make_pair(_secretingTypeName, secrOnContactData));
    }


    secrTypesNameSet.insert("SecretionOnContact");

}

void SecretionData::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {




    //emptying all containers
    typeNameSecrConstMap.clear();
    typeIdUptakeDataMap.clear();
    uptakeDataSet.clear();
    typeIdSecrConstMap.clear();
    secretionTypeNames.clear();
    secretionOnContactTypeNames.clear();
    constantConcentrationTypeNames.clear();
    secretionTypeIds.clear();
    secretionOnContactTypeIds.clear();
    constantConcentrationTypeIds.clear();
    typeIdSecrOnContactDataMap.clear();
    typeNameSecrConstMap.clear();
    typeNameSecrOnContactDataMap.clear();

    CC3DXMLElementList secrOnContactXMLVec = _xmlData->getElements("SecretionOnContact");
    for (unsigned int i = 0; i < secrOnContactXMLVec.size(); ++i) {
        string secreteType;
        float secrConst;
        unsigned char typeId;
        unsigned char contactTypeId;
        vector <string> contactTypeNamesVec;
        string contactTypeName;
        std::map<std::string, SecretionOnContactData>::iterator mitr;

        secreteType = secrOnContactXMLVec[i]->getAttribute("Type");
        //          typeId=automaton->getTypeId(secreteType);

        secretionOnContactTypeNames.insert(secreteType);

        if (secrOnContactXMLVec[i]->findAttribute("SecreteOnContactWith")) {
            contactTypeName = secrOnContactXMLVec[i]->getAttribute("SecreteOnContactWith");
            //parseStringIntoList(contactTypeNames,contactTypeNamesVec,",");

        }

        //          contactTypeId=automaton->getTypeId(contactTypeName);


        secrConst = secrOnContactXMLVec[i]->getDouble();


        //          mitr=typeIdSecrOnContactDataMap.find(typeId);
        mitr = typeNameSecrOnContactDataMap.find(secreteType);

        if (mitr != typeNameSecrOnContactDataMap.end()) {
            mitr->second.contactCellMapTypeNames.insert(make_pair(contactTypeName, secrConst));
        } else {
            SecretionOnContactData secrOnContactData;
            secrOnContactData.contactCellMapTypeNames.insert(make_pair(contactTypeName, secrConst));
            typeNameSecrOnContactDataMap.insert(make_pair(secreteType, secrOnContactData));

        }


        secrTypesNameSet.insert("SecretionOnContact");

        //break; //only one secretion Data allowed
    }
    CC3DXMLElementList secrXMLVec = _xmlData->getElements("Secretion");
    for (unsigned int i = 0; i < secrXMLVec.size(); ++i) {
        string secreteType;
        float secrConst;
        unsigned char typeId;
        secreteType = secrXMLVec[i]->getAttribute("Type");
        secretionTypeNames.insert(secreteType);
        //          typeId=automaton->getTypeId(secreteType);
        secrConst = secrXMLVec[i]->getDouble();

        //          typeIdSecrConstMap.insert(make_pair(typeId,secrConst));
        CC3D_Log(LOG_DEBUG) << "THIS IS secretrion type="<<secreteType<<" secrConst="<<secrConst;
        typeNameSecrConstMap.insert(make_pair(secreteType, secrConst));


        //secretionName="Secretion";
        secrTypesNameSet.insert("Secretion");

        //break; //only one secretion Data allowed

    }

    CC3DXMLElementList secrConstantConcentrationXMLVec = _xmlData->getElements("ConstantConcentration");
    for (unsigned int i = 0; i < secrConstantConcentrationXMLVec.size(); ++i) {
        string secreteType;
        float secrConst;
        unsigned char typeId;
        secreteType = secrConstantConcentrationXMLVec[i]->getAttribute("Type");
        constantConcentrationTypeNames.insert(secreteType);
        //          typeId=automaton->getTypeId(secreteType);
        secrConst = secrConstantConcentrationXMLVec[i]->getDouble();

        //          typeIdSecrConstMap.insert(make_pair(typeId,secrConst));

        //typeNameSecrConstMap.insert(make_pair(secreteType,secrConst));
        typeNameSecrConstConstantConcentrationMap.insert(make_pair(secreteType, secrConst));
        //secretionName="Secretion";
        secrTypesNameSet.insert("ConstantConcentration");

        //break; //only one secretion Data allowed

    }


    CC3DXMLElementList uptakeXMLVec = _xmlData->getElements("Uptake");
    for (unsigned int i = 0; i < uptakeXMLVec.size(); ++i) {
        string uptakeType;

        unsigned char typeId;
        uptakeType = uptakeXMLVec[i]->getAttribute("Type");

        UptakeData ud;
        if (uptakeXMLVec[i]->findAttribute("MaxUptake")) {
            ud.maxUptake = uptakeXMLVec[i]->getAttributeAsDouble("MaxUptake");
        }
        if (uptakeXMLVec[i]->findAttribute("RelativeUptakeRate")) {

            ud.relativeUptakeRate = uptakeXMLVec[i]->getAttributeAsDouble("RelativeUptakeRate");
        }

        if (uptakeXMLVec[i]->findAttribute("MichaelisMentenCoef")) {
            ud.mmCoef = uptakeXMLVec[i]->getAttributeAsDouble("MichaelisMentenCoef");
        }

        ud.typeName = uptakeType;


        uptakeDataSet.insert(ud);
        secrTypesNameSet.insert("Secretion");
    }


    active = true;


}

void SecretionData::initialize(Automaton *_automaton) {

    typeIdSecrConstMap.clear();
    for (std::map<std::string, float>::iterator mitr = typeNameSecrConstMap.begin();
         mitr != typeNameSecrConstMap.end(); ++mitr) {
        unsigned char typeId = _automaton->getTypeId(mitr->first);

        typeIdSecrConstMap.insert(make_pair(typeId, mitr->second));
    }

    typeIdSecrConstConstantConcentrationMap.clear();

    for (std::map<std::string, float>::iterator mitr = typeNameSecrConstConstantConcentrationMap.begin();
         mitr != typeNameSecrConstConstantConcentrationMap.end(); ++mitr) {
        unsigned char typeId = _automaton->getTypeId(mitr->first);

        typeIdSecrConstConstantConcentrationMap.insert(make_pair(typeId, mitr->second));
    }


    typeIdSecrOnContactDataMap.clear();

    for (std::map<std::string, SecretionOnContactData>::iterator mitr = typeNameSecrOnContactDataMap.begin();
         mitr != typeNameSecrOnContactDataMap.end(); ++mitr) {
        unsigned char typeId = _automaton->getTypeId(mitr->first);
        SecretionOnContactData &secretionOnContactData = mitr->second;

        //translating type name to typeId in SecretionOnContactData
        for (std::map<std::string, float>::iterator mitrSF = secretionOnContactData.contactCellMapTypeNames.begin();
             mitrSF != secretionOnContactData.contactCellMapTypeNames.end(); ++mitrSF) {
            unsigned char typeIdSOCD = _automaton->getTypeId(mitrSF->first);
            secretionOnContactData.contactCellMap.insert(make_pair(typeIdSOCD, mitrSF->second));
        }

        typeIdSecrOnContactDataMap.insert(make_pair(typeId, secretionOnContactData));

    }

    for (std::set<std::string>::iterator sitr = secretionTypeNames.begin(); sitr != secretionTypeNames.end(); ++sitr) {
        secretionTypeIds.insert(_automaton->getTypeId(*sitr));
    }

    for (std::set<std::string>::iterator sitr = secretionOnContactTypeNames.begin();
         sitr != secretionOnContactTypeNames.end(); ++sitr) {
        secretionOnContactTypeIds.insert(_automaton->getTypeId(*sitr));
    }

    for (std::set<std::string>::iterator sitr = constantConcentrationTypeNames.begin();
         sitr != constantConcentrationTypeNames.end(); ++sitr) {
        constantConcentrationTypeIds.insert(_automaton->getTypeId(*sitr));
    }


    //uptake
    for (std::set<UptakeData>::iterator sitr = uptakeDataSet.begin(); sitr != uptakeDataSet.end(); ++sitr) {
// 			  (const_cast<UptakeData>(*sitr)).typeId=_automaton->getTypeId(sitr->typeName);
        UptakeData ud = *sitr;
        ud.typeId = _automaton->getTypeId(sitr->typeName);

        typeIdUptakeDataMap.insert(make_pair(ud.typeId, ud));
    }

}

