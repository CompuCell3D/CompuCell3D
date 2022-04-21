#ifndef DIFFSECRDATA_H
#define DIFFSECRDATA_H


#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <limits>
#include <climits>

#undef max
#undef min

#include <CompuCell3D/SteerableObject.h>

#include "PDESolversDLLSpecifier.h"


namespace CompuCell3D {
    class Automaton;

    class PDESOLVERS_EXPORT SecretionOnContactData {
    public:
        std::map<unsigned char, float> contactCellMap;

        std::map<std::string, float> contactCellMapTypeNames;
    };

    class PDESOLVERS_EXPORT UptakeData {
    public:
        //enum UptakeType{RELATIVE,MM};
        UptakeData() : typeId(0), maxUptake(0.0), relativeUptakeRate(0.0), mmCoef(0.0) {}

        std::string typeName;
        unsigned char typeId;
        float maxUptake;
        float relativeUptakeRate;
        float mmCoef;//MichaelisMEnten Coeff - for Michaelis Menten type of uptake curve
        //UptakeType uptakeType;
        bool operator<(const UptakeData &rhs) const {
            return typeName < rhs.typeName;
        }


    };


    class PDESOLVERS_EXPORT SecretionData : public SteerableObject {
    protected:
        Automaton *automaton;
    public:
        SecretionData() :
                active(false) {}

        enum SecretionMode {
            SECRETION = 10001, SECRETION_ON_CONTACT = 10002, CONSTANT_CONCENTRATION = 10003
        };

        //SecretionMode secrMode;
        void setAutomaton(Automaton *_automaton) { automaton = _automaton; }

        bool active;
        std::set <std::string> secrTypesNameSet;

        //uptake
        std::map<unsigned char, UptakeData> typeIdUptakeDataMap;
        std::set <UptakeData> uptakeDataSet;


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


        std::map<unsigned char, SecretionOnContactData> typeIdSecrOnContactDataMap;

        std::map<std::string, float> typeNameSecrConstMap;
        std::map<std::string, float> typeNameSecrConstConstantConcentrationMap;

        std::map <std::string, SecretionOnContactData> typeNameSecrOnContactDataMap;

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


    };

    class PDESOLVERS_EXPORT CouplingData {
    public:
        CouplingData(std::string _intrFieldName = "", unsigned int _fieldIdx = 0, float _couplingCoef = 0.0) :
                intrFieldName(_intrFieldName),
                fieldIdx(_fieldIdx),
                couplingCoef(_couplingCoef) {
        }

        std::string intrFieldName;
        unsigned int fieldIdx;
        float couplingCoef;

    };


    class PDESOLVERS_EXPORT DiffusionData : public SteerableObject {
        Automaton *automaton;
    public:
        DiffusionData() :
                active(false),
                diffConst(0.0),
                decayConst(0.0),
                deltaX(1.0),
                deltaT(1.0),
                useThresholds(false),
                maxConcentration(std::numeric_limits<float>::max()),
                minConcentration(std::numeric_limits<float>::min()),
                useBoxWatcher(false),
                extraTimesPerMCS(0),
                variableDiffusionCoefficientFlag(false) {
            for (int i = 0; i < UCHAR_MAX + 1; ++i) {
                decayCoef[i] = 0.0;
                diffCoef[i] = 0.0;
            }
        }

        void setAutomaton(Automaton *_automaton) { automaton = _automaton; }

        bool active;
        float diffConst;
        float decayConst;
        // type dependent decay and diff coefs
        float decayCoef[UCHAR_MAX + 1];
        float diffCoef[UCHAR_MAX + 1];

        float deltaX;
        float deltaT;
        float dt_dx2;


        bool useThresholds;
        float maxConcentration;
        float minConcentration;
        bool useBoxWatcher;
        int extraTimesPerMCS;
        std::string diffusionName;
        std::vector <CouplingData> couplingDataVec;
        std::string additionalTerm;

        //steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        void initialize(Automaton *_automaton);

        //bool getVariableDiffusionCoeeficientFlag();
        bool getVariableDiffusionCoeeficientFlag();

        void DoNotDiffuseTo(std::string _typeName);

        void DoNotDecayIn(std::string _typeName);

        void FieldName(std::string _fieldName);

        void ConcentrationFileName(std::string _concFieldName);

        void DiffusionConstant(float _diffConst);

        void DecayConstant(float _decayConst);

        void DeltaX(float _dx);

        void DeltaT(float _dt);

        void UseBoxWatcher(bool _flag);

        void CouplingTerm(std::string _interactingFieldName, float _couplingCoefficient);

        void MinConcentrationThreshold(float _min);

        void MaxConcentrationThreshold(float _max);


        std::string fieldName;
        std::string concentrationFileName;
        std::string initialConcentrationExpression;

        //std::vector<std::string> avoidTypeVec;
        std::set<unsigned char> avoidTypeIdSet;
        std::set<unsigned char> avoidDecayInIdSet;

        std::set <std::string> avoidTypeNameSet;
        std::set <std::string> avoidDecayInTypeNameSet;

        std::map<std::string, float> diffCoefTypeNameMap;
        std::map<std::string, float> decayCoefTypeNameMap;

        bool variableDiffusionCoefficientFlag;

        int userFuncFlag;
        std::string FieldDependenciesSTR;
        std::vector <std::string> fieldDependencies;
        std::string funcName;


        friend std::ostream &operator<<(std::ostream &out, CompuCell3D::DiffusionData &diffData);
    };


    inline std::ostream &operator<<(std::ostream &out, CompuCell3D::DiffusionData &diffData) {
        using namespace std;

        out << "DiffusionConstant: " << diffData.diffConst << endl;
        out << "DecayConstant: " << diffData.decayConst << endl;
        out << "DeltaX: " << diffData.deltaX << endl;
        out << "DeltaT: " << diffData.deltaT << endl;
        out << "fieldName: " << diffData.fieldName << endl;

        return out;
    }


};

#endif
