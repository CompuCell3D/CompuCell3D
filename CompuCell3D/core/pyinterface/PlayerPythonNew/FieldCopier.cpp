//
// Created by m on 3/22/25.
//

#include "FieldCopier.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>

using namespace CompuCell3D;

#define DISPATCH_ENTRY(TYPE) \
    { typeid(TYPE), [this, &attribute_name](void* ptr) { \
        auto typedPtr = static_cast<Field3D<TYPE>*>(ptr); \
        auto& extractorMap = getAttributeExtractorMap<TYPE>(); \
        auto it = extractorMap.find(attribute_name); \
        return it != extractorMap.end() ? fillCellAttributeValues<TYPE>(typedPtr, it->second) : false; \
    }}


template<typename T>
const std::unordered_map<std::string, std::function<T(CellG*)>>& getAttributeExtractorMap() {
    // CellG scalar attributes
    static const std::unordered_map<std::string, std::function<T(CellG*)>> extractorMap = {
            {"type",      [](CellG *cell) { return static_cast<T>(cell->type); }},
            {"id",        [](CellG *cell) { return static_cast<T>(cell->id); }},
            {"clusterId", [](CellG *cell) { return static_cast<T>(cell->clusterId); }},
            {"volume", [](CellG *cell) { return static_cast<T>(cell->volume); }},
            {"lambdaVolume", [](CellG *cell) { return static_cast<T>(cell->lambdaVolume); }},
            {"targetVolume", [](CellG *cell) { return static_cast<T>(cell->targetVolume); }},
            {"surface", [](CellG *cell) { return static_cast<T>(cell->surface); }},
            {"lambdaSurface", [](CellG *cell) { return static_cast<T>(cell->lambdaSurface); }},
            {"targetSurface", [](CellG *cell) { return static_cast<T>(cell->targetSurface); }},
            {"clusterSurface", [](CellG *cell) { return static_cast<T>(cell->clusterSurface); }},
            {"lambdaClusterSurface", [](CellG *cell) { return static_cast<T>(cell->lambdaClusterSurface); }},
            {"targetClusterSurface", [](CellG *cell) { return static_cast<T>(cell->targetClusterSurface); }},
            {"ecc", [](CellG *cell) { return static_cast<T>(cell->ecc); }},
            {"connectivityOn", [](CellG *cell) { return static_cast<T>(cell->connectivityOn); }},
            
    };
    return extractorMap;
}


FieldCopier::FieldCopier(Simulator *sim):sim(sim), potts(nullptr) {
    if (sim){
        potts = sim->getPotts();
    }
    initializeCoreCopierFunctionMap();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldCopier::initializeCoreCopierFunctionMap(){
    coreCopierFunctionMap = {
            {typeid(char), [this](void* ptr) {
                return this->fillCellTypeValues<char>( static_cast<Field3D<char>*>(ptr));
            }},

    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<std::type_index, void*> FieldCopier::getFieldTypeAndPointer( const std::string& fieldName) {
    Field3DTypeBase *conFieldBasePtr = sim->getGenericScalarFieldTypeBase(fieldName);


    static const std::unordered_map<std::type_index, std::function<void *(Field3DTypeBase *)>> typeCaster = {
            {typeid(char),               [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<char> *>(base)); }},
            {typeid(unsigned char),      [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned char> *>(base)); }},
            {typeid(short),              [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<short> *>(base)); }},
            {typeid(unsigned short),     [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned short> *>(base)); }},
            {typeid(int),                [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<int> *>(base)); }},
            {typeid(unsigned int),       [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned int> *>(base)); }},
            {typeid(long),               [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<long> *>(base)); }},
            {typeid(unsigned long),      [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned long> *>(base)); }},
            {typeid(long long),          [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<long long> *>(base)); }},
            {typeid(unsigned long long), [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<unsigned long long> *>(base)); }},
            {typeid(float),              [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<float> *>(base)); }},
            {typeid(double),             [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<double> *>(base)); }},
            {typeid(long double),        [](
                    Field3DTypeBase *base) { return static_cast<void *>(dynamic_cast<NumpyArrayWrapper3DImpl<long double> *>(base)); }},
    };
    if (conFieldBasePtr) {
        std::type_index fieldType = conFieldBasePtr->getType();
        auto it = typeCaster.find(fieldType);

        if (it != typeCaster.end()) {
            return {fieldType, it->second(conFieldBasePtr)};
        }

    }

    Field3D<float> *conFieldPtr = nullptr;
    std::map<std::string, Field3D<float> *> &fieldMap = sim->getConcentrationFieldNameMap();
    std::map<std::string, Field3D<float> *>::iterator mitr;
    mitr = fieldMap.find(fieldName);
    if (mitr != fieldMap.end()) {
        conFieldPtr = mitr->second;
    }

    if (conFieldPtr) {

        return {std::type_index(typeid(float)), conFieldPtr};
    }


    return {std::type_index(typeid(void)), nullptr};  // Unsupported type

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool FieldCopier::copy_cell_type_field_values_to(const std::string& field_name){
    // Retrieve field type and pointer
    auto result = getFieldTypeAndPointer(field_name);
    std::type_index fieldType = std::get<0>(result);
    void* fieldPtr = std::get<1>(result);

    if (!fieldPtr || fieldType == typeid(void)) {
        return false;
    }


    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
bool FieldCopier::fillCellTypeValues( Field3D<T> *fieldPtr){

    if (!fieldPtr) return false;

    Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
    Dim3D fieldDim = cellFieldG->getDim();
    Point3D pt;
    for (pt.z = 0; pt.z < fieldDim.z ; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y ; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x ; ++pt.x){
                auto cell = cellFieldG->get(pt);

                fieldPtr->set(pt, cell ? cell->type: 0);
            }
    return true;
}

template<typename T>
bool FieldCopier::fillCellAttributeValues(Field3D<T> *fieldPtr, std::function<T(CellG*)> extractor) {
    
    if (!fieldPtr) return false;

    Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
    Dim3D fieldDim = cellFieldG->getDim();

    Point3D pt;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {
                auto cell = cellFieldG->get(pt);

                fieldPtr->set(pt, cell ? extractor(cell) : T(0));
            }
    return true;
}

bool FieldCopier::copy_cell_attribute_field_values_to(const std::string& field_name, const std::string& attribute_name) {
    auto result = getFieldTypeAndPointer(field_name);
    std::type_index fieldType = std::get<0>(result);
    void* fieldPtr = std::get<1>(result);

    if (!fieldPtr || fieldType == typeid(void)) {
        return false;
    }

    using DispatchFn = std::function<bool(void*)>;
    static const std::unordered_map<std::type_index, DispatchFn> dispatchMap = {
            DISPATCH_ENTRY(char),
            DISPATCH_ENTRY(unsigned char),
            DISPATCH_ENTRY(short),
            DISPATCH_ENTRY(unsigned short),
            DISPATCH_ENTRY(int),
            DISPATCH_ENTRY(unsigned int),
            DISPATCH_ENTRY(long),
            DISPATCH_ENTRY(unsigned long),
            DISPATCH_ENTRY(long long),
            DISPATCH_ENTRY(unsigned long long),
            DISPATCH_ENTRY(float),
            DISPATCH_ENTRY(double),
            DISPATCH_ENTRY(long double),
    };
//    static const std::unordered_map<std::type_index, DispatchFn> dispatchMap = {
//            {typeid(char),               [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<char>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<char>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<char>(typedPtr, it->second) : false;
//            }},
//            {typeid(unsigned char),      [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<unsigned char>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<unsigned char>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<unsigned char>(typedPtr, it->second) : false;
//            }},
//            {typeid(short),              [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<short>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<short>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<short>(typedPtr, it->second) : false;
//            }},
//            {typeid(unsigned short),     [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<unsigned short>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<unsigned short>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<unsigned short>(typedPtr, it->second) : false;
//            }},
//            {typeid(int),                [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<int>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<int>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<int>(typedPtr, it->second) : false;
//            }},
//            {typeid(unsigned int),       [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<unsigned int>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<unsigned int>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<unsigned int>(typedPtr, it->second) : false;
//            }},
//            {typeid(long),               [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<long>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<long>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<long>(typedPtr, it->second) : false;
//            }},
//            {typeid(unsigned long),      [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<unsigned long>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<unsigned long>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<unsigned long>(typedPtr, it->second) : false;
//            }},
//            {typeid(long long),          [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<long long>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<long long>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<long long>(typedPtr, it->second) : false;
//            }},
//            {typeid(unsigned long long), [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<unsigned long long>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<unsigned long long>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<unsigned long long>(typedPtr, it->second) : false;
//            }},
//            {typeid(float),              [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<float>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<float>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<float>(typedPtr, it->second) : false;
//            }},
//            {typeid(double),             [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<double>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<double>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<double>(typedPtr, it->second) : false;
//            }},
//            {typeid(long double),        [this, &attribute_name](void* ptr) {
//                auto typedPtr = static_cast<Field3D<long double>*>(ptr);
//                auto& extractorMap = getAttributeExtractorMap<long double>();
//                auto it = extractorMap.find(attribute_name);
//                return it != extractorMap.end() ? fillCellAttributeValues<long double>(typedPtr, it->second) : false;
//            }},
//    };

    auto it = dispatchMap.find(fieldType);

    if (it != dispatchMap.end()) {

        return it->second(fieldPtr);
    }

    return false;
}



