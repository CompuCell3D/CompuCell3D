//
// Created by m on 2/1/25.
// Base class used to facilitate storagte and retrieval fields with different underlying type from tracking containers

#include <typeindex>
#include <typeinfo>
#include <unordered_map>


#ifndef COMPUCELL3D_FIELD3DTYPEBASE_H
#define COMPUCELL3D_FIELD3DTYPEBASE_H

namespace CompuCell3D {

    class Field3DTypeBase {
    public:
        // needed for proper deletion of fields stored via pointer to Field3DTypeBase in e.g. map <str, unique_ptr<Field3DTypeBase> >
        virtual ~Field3DTypeBase() = default;
        virtual void displayType() const = 0;
        virtual std::string getTypeString() const = 0;
        // Get type information - using type_index
        virtual const std::type_index& getType() const = 0;
        virtual std::string getNumPyTypeString() const = 0;
        // Function to map typeid(T).name() to NumPy type
        static std::string getNumPyType(const std::type_info& typeInfo) {
            static const std::unordered_map<std::string, std::string> typeMap = {
                    {typeid(char).name(), "int8"},
                    {typeid(unsigned char).name(), "uint8"},
                    {typeid(short).name(), "int16"},
                    {typeid(unsigned short).name(), "uint16"},
                    {typeid(int).name(), "int32"},
                    {typeid(unsigned int).name(), "uint32"},
                    {typeid(long).name(), "int64"},
                    {typeid(unsigned long).name(), "uint64"},
                    {typeid(long long).name(), "int64"},
                    {typeid(unsigned long long).name(), "uint64"},
                    {typeid(float).name(), "float32"},
                    {typeid(double).name(), "float64"},
                    {typeid(long double).name(), "float128"}
            };

            auto it = typeMap.find(typeInfo.name());
            if (it != typeMap.end()) {
                return it->second;
            } else {
                return "np.object"; // Default to generic NumPy object type
            }
        }

    };

};

#endif //COMPUCELL3D_FIELD3DTYPEBASE_H
