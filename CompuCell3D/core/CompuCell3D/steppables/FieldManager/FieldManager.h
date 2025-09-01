

#ifndef FIELDMANAGERSTEPPABLE_H
#define FIELDMANAGERSTEPPABLE_H

#include <CompuCell3D/CC3D.h>



#include "FieldManagerDLLSpecifier.h"

namespace CompuCell3D {
    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class CellInventory;
    class CellG;

    class FIELDMANAGER_EXPORT FieldSpec {
    public:
        // Enum for field type
        enum class FieldType {
            Scalar,
            Vector,
//            ScalarDouble,
//            ScalarChar,
//            ScalarUChar,
//            ScalarShort,
//            ScalarUShort,
//            ScalarInt,
//            ScalarUInt,
//            ScalarLong,
//            ScalarULong

        };

        enum class PrecisionType {
            Float,
            Double,
            Char,
            UChar,
            Short,
            UShort,
            Int,
            UInt,
            Long,
            ULong
        };

        // Enum for field kind
        enum class FieldKind {
            NumPy,
            CC3D
        };

        // Constructor
        FieldSpec(){}

        // Member variables
        std::string name; // Name of the field
        int padding=1;      // Padding for the field
        FieldType type = FieldType::Scalar;        // Type of the field (scalar or vector)
        FieldKind kind = FieldKind::NumPy;        // Kind of the field (NumPy or CC3D)
        PrecisionType precision = PrecisionType::Float;

        // Function to map string to Kind
        static FieldKind mapStringToKind(const std::string& kindStr) {
            // Convert input string to lowercase for case-insensitivity
            std::string lowerKindStr = kindStr;
            std::transform(lowerKindStr.begin(), lowerKindStr.end(), lowerKindStr.begin(), ::tolower);

            // Unordered map for string-to-enum mapping
            static const std::unordered_map<std::string, FieldKind> kindMap = {
                    {"numpy", FieldKind::NumPy},
                    {"cc3d", FieldKind::CC3D}
            };

            // Look up the string in the map
            auto it = kindMap.find(lowerKindStr);
            if (it != kindMap.end()) {
                return it->second;
            } else {
                ASSERT_OR_THROW("Invalid kind: " + kindStr, false)
            }
        }

        // Function to map string to Type
        static FieldType mapStringToType(const std::string& typeStr) {
            // Convert input string to lowercase for case-insensitivity
            std::string lowerTypeStr = typeStr;
            std::transform(lowerTypeStr.begin(), lowerTypeStr.end(), lowerTypeStr.begin(), ::tolower);

            // Unordered map for string-to-enum mapping
            static const std::unordered_map<std::string, FieldType> typeMap = {
                    {"scalar", FieldType::Scalar},
                    {"concentration", FieldType::Scalar},
                    {"vector", FieldType::Vector},
            };

            // Look up the string in the map
            auto it = typeMap.find(lowerTypeStr);
            if (it != typeMap.end()) {
                return it->second;
            } else {
                ASSERT_OR_THROW("Invalid type: " + typeStr, false)

            }
        }

        static PrecisionType mapStringToPrecision(const std::string& precisionStr){
            std::string lowerPrecisionStr = precisionStr;
            std::transform(lowerPrecisionStr.begin(), lowerPrecisionStr.end(), lowerPrecisionStr.begin(), ::tolower);

            static const std::unordered_map<std::string, PrecisionType> precisionMap = {
                    // C++ style
                    {"float", PrecisionType::Float},
                    {"double", PrecisionType::Double},
                    {"char", PrecisionType::Char},
                    {"uchar", PrecisionType::UChar},
                    {"short", PrecisionType::Short},
                    {"ushort", PrecisionType::UShort},
                    {"int", PrecisionType::Int},
                    {"uint", PrecisionType::UInt},
                    {"long", PrecisionType::Long},
                    {"ulong", PrecisionType::ULong},

                    // NumPy-style
                    {"float32", PrecisionType::Float},
                    {"float64", PrecisionType::Double},
                    {"int8", PrecisionType::Char},
                    {"uint8", PrecisionType::UChar},
                    {"int16", PrecisionType::Short},
                    {"uint16", PrecisionType::UShort},
                    {"int32", PrecisionType::Int},
                    {"uint32", PrecisionType::UInt},
                    {"int64", PrecisionType::Long},
                    {"uint64", PrecisionType::ULong}
            };

            auto it = precisionMap.find(lowerPrecisionStr);
            if (it != precisionMap.end()) {
                return it->second;
            } else {
                ASSERT_OR_THROW("Invalid precision: " + precisionStr, false)
            }
        }

        // Utility method to print field information
        void printInfo() const {
            std::cout << "Field Name: " << name << "\n"
                      << "Padding: " << padding << "\n"
                      << "Type: " << (type == FieldType::Scalar ? "Scalar" : "Vector") << "\n"
                      << "Kind: " << (kind == FieldKind::NumPy ? "NumPy" : "CC3D") << "\n";
        }
    };

  class FIELDMANAGER_EXPORT FieldManager : public Steppable {

                    

    WatchableField3D<CellG *> *cellFieldG;

    Simulator * sim;
    Potts3D *potts;
    CC3DXMLElement *xmlData;
    Automaton *automaton;
    BoundaryStrategy *boundaryStrategy;
    CellInventory * cellInventoryPtr;
    Dim3D fieldDim;
    std::vector<FieldSpec> fieldSpecVec;

  public:

    FieldManager ();
    virtual ~FieldManager ();

    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual void extraInit(Simulator *simulator);
    

    //steppable interface
    virtual void start();
    virtual void step(const unsigned int currentStep);
    virtual void finish() {}

    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	virtual std::string toString();

  };

};

#endif        

