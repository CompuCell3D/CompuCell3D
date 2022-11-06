
#include <CompuCell3D/CC3D.h>


using namespace CompuCell3D;
using namespace std;

#include "SecretionPlugin.h"
#include "SecretionDataP.h"

#include <Logger/CC3DLogger.h>


std::string SecretionDataP::steerableName(){
	return "SecretionDataP";
}


void SecretionDataP::Secretion(std::string _typeName, float _secretionConst) {
    typeNameSecrConstMap.insert(make_pair(_typeName, _secretionConst));
    secrTypesNameSet.insert("Secretion");
}

void SecretionDataP::SecretionOnContact(std::string _secretingTypeName, std::string _onContactWithTypeName,
                                        float _secretionConst) {
    std::map<std::string, SecretionOnContactDataP>::iterator mitr;
    mitr = typeNameSecrOnContactDataMap.find(_secretingTypeName);

    if (mitr != typeNameSecrOnContactDataMap.end()) {
        mitr->second.contactCellMapTypeNames.insert(make_pair(_onContactWithTypeName, _secretionConst));
    } else {
        SecretionOnContactDataP secrOnContactData;
        secrOnContactData.contactCellMapTypeNames.insert(make_pair(_onContactWithTypeName, _secretionConst));
        typeNameSecrOnContactDataMap.insert(make_pair(_secretingTypeName, secrOnContactData));
    }


    secrTypesNameSet.insert("SecretionOnContact");

}

void SecretionDataP::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {




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


    fieldName = _xmlData->getAttribute("Name");
    if (_xmlData->findAttribute("ExtraTimesPerMC")) {
        timesPerMCS += _xmlData->getAttributeAsUInt("ExtraTimesPerMC");
    }

    if (_xmlData->findElement("UseBoxWatcher"))
        useBoxWatcher = true;


    CC3DXMLElementList secrOnContactXMLVec = _xmlData->getElements("SecretionOnContact");
    for (unsigned int i = 0; i < secrOnContactXMLVec.size(); ++i) {
        string secreteType;
        float secrConst;
        unsigned char typeId;
        unsigned char contactTypeId;
        vector <string> contactTypeNamesVec;
        string contactTypeName;
        std::map<std::string, SecretionOnContactDataP>::iterator mitr;

        secreteType = secrOnContactXMLVec[i]->getAttribute("Type");

        secretionOnContactTypeNames.insert(secreteType);

        if (secrOnContactXMLVec[i]->findAttribute("SecreteOnContactWith")) {
            contactTypeName = secrOnContactXMLVec[i]->getAttribute("SecreteOnContactWith");


        }

        secrConst = secrOnContactXMLVec[i]->getDouble();

        mitr = typeNameSecrOnContactDataMap.find(secreteType);

        if (mitr != typeNameSecrOnContactDataMap.end()) {
            mitr->second.contactCellMapTypeNames.insert(make_pair(contactTypeName, secrConst));
        } else {
            SecretionOnContactDataP secrOnContactData;
            secrOnContactData.contactCellMapTypeNames.insert(make_pair(contactTypeName, secrConst));
            typeNameSecrOnContactDataMap.insert(make_pair(secreteType, secrOnContactData));

        }


        secrTypesNameSet.insert("SecretionOnContact");


    }
    CC3DXMLElementList secrXMLVec = _xmlData->getElements("Secretion");
    for (unsigned int i = 0; i < secrXMLVec.size(); ++i) {
        string secreteType;

        float secrConst;
        unsigned char typeId;
        secreteType = secrXMLVec[i]->getAttribute("Type");
        secretionTypeNames.insert(secreteType);

        secrConst = secrXMLVec[i]->getDouble();

        CC3D_Log(LOG_DEBUG) << "THIS IS secretrion type=" << secreteType << " secrConst=" << secrConst;
        typeNameSecrConstMap.insert(make_pair(secreteType, secrConst));


        secrTypesNameSet.insert("Secretion");

    }


    CC3DXMLElementList secrConstantConcentrationXMLVec = _xmlData->getElements("ConstantConcentration");
    for (unsigned int i = 0; i < secrConstantConcentrationXMLVec.size(); ++i) {
        string secreteType;
        float secrConst;
        unsigned char typeId;
        secreteType = secrConstantConcentrationXMLVec[i]->getAttribute("Type");
        constantConcentrationTypeNames.insert(secreteType);

        secrConst = secrConstantConcentrationXMLVec[i]->getDouble();

        typeNameSecrConstConstantConcentrationMap.insert(make_pair(secreteType, secrConst));

        secrTypesNameSet.insert("ConstantConcentration");

    }


    CC3DXMLElementList uptakeXMLVec = _xmlData->getElements("Uptake");
    for (unsigned int i = 0; i < uptakeXMLVec.size(); ++i) {
        string uptakeType;
        float maxUptake;
        float relativeUptakeRate;

        unsigned char typeId;
        uptakeType = uptakeXMLVec[i]->getAttribute("Type");


        maxUptake = uptakeXMLVec[i]->getAttributeAsDouble("MaxUptake");
        relativeUptakeRate = uptakeXMLVec[i]->getAttributeAsDouble("RelativeUptakeRate");


        UptakeDataP ud;
        ud.typeName = uptakeType;
        ud.maxUptake = maxUptake;
        ud.relativeUptakeRate = relativeUptakeRate;

        uptakeDataSet.insert(ud);
        secrTypesNameSet.insert("Secretion");
    }


    active = true;

    secretionFcnPtrVec.assign(secrTypesNameSet.size(), 0);
    unsigned int j = 0;
    for (set<string>::iterator sitr = secrTypesNameSet.begin(); sitr != secrTypesNameSet.end(); ++sitr) {

        if ((*sitr) == "Secretion") {
            secretionFcnPtrVec[j] = &SecretionPlugin::secreteSingleField;
            ++j;
        } else if ((*sitr) == "SecretionOnContact") {
            secretionFcnPtrVec[j] = &SecretionPlugin::secreteOnContactSingleField;
            ++j;
        } else if ((*sitr) == "ConstantConcentration") {
            secretionFcnPtrVec[j] = &SecretionPlugin::secreteConstantConcentrationSingleField;
            ++j;
        }
    }


}

void SecretionDataP::initialize(Automaton *_automaton) {

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

    for (std::map<std::string, SecretionOnContactDataP>::iterator mitr = typeNameSecrOnContactDataMap.begin();
         mitr != typeNameSecrOnContactDataMap.end(); ++mitr) {
        unsigned char typeId = _automaton->getTypeId(mitr->first);
        SecretionOnContactDataP &secretionOnContactData = mitr->second;

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
    for (std::set<UptakeDataP>::iterator sitr = uptakeDataSet.begin(); sitr != uptakeDataSet.end(); ++sitr) {
        UptakeDataP ud = *sitr;
        ud.typeId = _automaton->getTypeId(sitr->typeName);

        typeIdUptakeDataMap.insert(make_pair(ud.typeId, ud));
    }

}

