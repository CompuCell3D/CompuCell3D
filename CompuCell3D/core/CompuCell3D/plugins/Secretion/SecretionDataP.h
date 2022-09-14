#ifndef SECRETIONDATAPPLUGIN_H
#define SECRETIONDATAPPLUGIN_H


#include <string>
#include <vector>
#include <map>
#include <set>


#include "SecretionDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

    class Potts3D;

    class CellG;

    class Steppable;

    class Simulator;

    class Automaton;

    class SecretionPlugin;


    class SECRETION_EXPORT SecretionOnContactDataP {
    public:
        std::map<unsigned char, float> contactCellMap;

        std::map<std::string, float> contactCellMapTypeNames;

    };

    class SECRETION_EXPORT UptakeDataP {
    public:
        UptakeDataP() : typeId(0), maxUptake(0.0), relativeUptakeRate(0.0) {}

        std::string typeName;
        unsigned char typeId;
        float maxUptake;
        float relativeUptakeRate;

        bool operator<(const UptakeDataP &rhs) const {
            return typeName < rhs.typeName;
        }


    };


    class SECRETION_EXPORT SecretionDataP : public SteerableObject {
    protected:
        Automaton *automaton;
    public:

        typedef void (SecretionPlugin::*secrSingleFieldFcnPtr_t)(unsigned int idx);

        SecretionDataP() :
                automaton(0),
                active(false),
                timesPerMCS(1),
                useBoxWatcher(false) {}

        enum SecretionMode {
            SECRETION = 10001, SECRETION_ON_CONTACT = 10002, CONSTANT_CONCENTRATION = 10003
        };

        //SecretionMode secrMode;
        void setAutomaton(Automaton *_automaton) { automaton = _automaton; }

        bool active;
        std::set <std::string> secrTypesNameSet;

        //uptake
        std::map<unsigned char, UptakeDataP> typeIdUptakeDataMap;
        std::set <UptakeDataP> uptakeDataSet;


        //std::string secretionName;

        std::map<unsigned char, float> typeIdSecrConstMap;
        std::map<unsigned char, float> typeIdSecrConstConstantConcentrationMap;

        //Here I store type Names of types secreting in a given mode
        std::set <std::string> secretionTypeNames;
        std::set <std::string> secretionOnContactTypeNames;
        std::set <std::string> constantConcentrationTypeNames;
        //Here I store type Names of types secreting in a given mode
        std::set<unsigned char> secretionTypeIds;
        std::set<unsigned char> secretionOnContactTypeIds;
        std::set<unsigned char> constantConcentrationTypeIds;


        std::map<unsigned char, SecretionOnContactDataP> typeIdSecrOnContactDataMap;

        std::map<std::string, float> typeNameSecrConstMap;
        std::map<std::string, float> typeNameSecrConstConstantConcentrationMap;

        std::map <std::string, SecretionOnContactDataP> typeNameSecrOnContactDataMap;

        float getSimpleSecretionConstByTypeName(std::string _typeName) {
            std::map<std::string, float>::iterator mitr = typeNameSecrConstMap.find(_typeName);
            if (mitr != typeNameSecrConstMap.end()) {
                return mitr->second;
            }
            return 0;
        }

        void setSimpleSecretionConstByTypeName(std::string _typeName, float _const) {
            std::map<std::string, float>::iterator mitr = typeNameSecrConstMap.find(_typeName);
            if (mitr != typeNameSecrConstMap.end()) {
                mitr->second = _const;
            } else {
                Secretion(_typeName, _const);
            }

        }


        void Secretion(std::string _typeName, float _secretionConst);

        void
        SecretionOnContact(std::string _secretingTypeName, std::string _onContactWithTypeName, float _secretionConst);

        //steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        void initialize(Automaton *_automaton);

        std::string fieldName;
        int timesPerMCS;
        bool useBoxWatcher;

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;


    };


};
#endif

