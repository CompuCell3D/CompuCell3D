//
// Created by m on 3/22/25.
//

#include "FieldCopier.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/CC3DExceptions.h>

using namespace CompuCell3D;

//#define DISPATCH_ENTRY(TYPE) \
//    { typeid(TYPE), [this, attribute_name](void* ptr) { \
//        auto typedPtr = static_cast<Field3D<TYPE>*>(ptr); \
//        auto& extractorMap = getAttributeExtractorMap<TYPE>(); \
//        auto it = extractorMap.find(attribute_name); \
//        return it != extractorMap.end() ? fillCellAttributeValues<TYPE>(typedPtr, it->second) : false; \
//    }}

//#define DISPATCH_ENTRY(TYPE) \
//    { typeid(TYPE), [this, attribute_name](void* ptr) { \
//        auto typedPtr = static_cast<Field3D<TYPE>*>(ptr); \
//        const auto& extractorMap = getAttributeExtractorMap<TYPE>(); \
//        auto it = extractorMap.find(attribute_name); \
//        return it != extractorMap.end() ? fillCellAttributeValues<TYPE>(typedPtr, it->second) : false; \
//    }}




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

template<typename T>
std::function<bool(void*)> make_dispatch_lambda(FieldCopier* copier, const std::string& attribute_name) {
    return [copier, attribute_name](void* ptr) {
        auto typedPtr = static_cast<Field3D<T>*>(ptr);
        const auto& extractorMap = getAttributeExtractorMap<T>();
        auto it = extractorMap.find(attribute_name);
        return it != extractorMap.end() ? copier->fillCellAttributeValues<T>(typedPtr, it->second) : false;
    };
}



FieldCopier::FieldCopier(Simulator *sim):sim(sim), potts(nullptr) {
    if (sim){
        potts = sim->getPotts();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldCopier::get_available_attributes(){
    const auto& extractorMap = getAttributeExtractorMap<char>();
    std::vector<std::string> keys;
    keys.reserve(extractorMap.size());
    for (const auto& pair : extractorMap) {
        keys.push_back(pair.first);
    }
    return keys;
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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//bool FieldCopier::copy_cell_type_field_values_to(const std::string& field_name){
//    // Retrieve field type and pointer
//    auto result = getFieldTypeAndPointer(field_name);
//    std::type_index fieldType = std::get<0>(result);
//    void* fieldPtr = std::get<1>(result);
//
//    if (!fieldPtr || fieldType == typeid(void)) {
//        return false;
//    }
//
//
//    return true;
//}

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
    cerr<<"fieldPtr"<<fieldPtr<<endl;
    if (!fieldPtr) return false;

    Field3D<CellG *> *cellFieldG = potts->getCellFieldG();
    Dim3D fieldDim = cellFieldG->getDim();

    Point3D pt;
    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {
                auto cell = cellFieldG->get(pt);
                if (pt.y>=45 && pt.y<50 && pt.x>=45 && pt.x<50) {
                    cerr << "pt=" << pt << "val" << (T) (cell ? extractor(cell) : T(0)) << endl;
                }
                fieldPtr->set(pt, cell ? extractor(cell) : T(0));
            }
    return true;
}


bool FieldCopier::copy_cell_attribute_field_values_to(const std::string& field_name, const std::string& attribute_name) {
    auto result = getFieldTypeAndPointer(field_name);
    std::type_index fieldType = std::get<0>(result);
    void* fieldPtr = std::get<1>(result);

    if (!fieldPtr || fieldType == typeid(void)) {
        ASSERT_OR_THROW("Field " + field_name + " cannot be found", false);
    }

    using DispatchFn = std::function<bool(void*)>;
    const std::unordered_map<std::type_index, DispatchFn> dispatchMap = {
            {typeid(char),               make_dispatch_lambda<char>(this, attribute_name)},
            {typeid(unsigned char),      make_dispatch_lambda<unsigned char>(this, attribute_name)},
            {typeid(short),              make_dispatch_lambda<short>(this, attribute_name)},
            {typeid(unsigned short),     make_dispatch_lambda<unsigned short>(this, attribute_name)},
            {typeid(int),                make_dispatch_lambda<int>(this, attribute_name)},
            {typeid(unsigned int),       make_dispatch_lambda<unsigned int>(this, attribute_name)},
            {typeid(long),               make_dispatch_lambda<long>(this, attribute_name)},
            {typeid(unsigned long),      make_dispatch_lambda<unsigned long>(this, attribute_name)},
            {typeid(long long),          make_dispatch_lambda<long long>(this, attribute_name)},
            {typeid(unsigned long long), make_dispatch_lambda<unsigned long long>(this, attribute_name)},
            {typeid(float),              make_dispatch_lambda<float>(this, attribute_name)},
            {typeid(double),             make_dispatch_lambda<double>(this, attribute_name)},
            {typeid(long double),        make_dispatch_lambda<long double>(this, attribute_name)},
    };

    auto it = dispatchMap.find(fieldType);
    if (it != dispatchMap.end()) {
        return it->second(fieldPtr);
    }

    return false;
}

//bool FieldCopier::copy_cell_attribute_field_values_to(const std::string& field_name, const std::string& attribute_name) {
//    auto result = getFieldTypeAndPointer(field_name);
//    std::type_index fieldType = std::get<0>(result);
//    void* fieldPtr = std::get<1>(result);
//
//    if (!fieldPtr || fieldType == typeid(void)) {
//        ASSERT_OR_THROW("Field "+field_name+" cannot be found", false);
//    }
//
//    using DispatchFn = std::function<bool(void*)>;
//    static const std::unordered_map<std::type_index, DispatchFn> dispatchMap = {
//            DISPATCH_ENTRY(char),
//            DISPATCH_ENTRY(unsigned char),
//            DISPATCH_ENTRY(short),
//            DISPATCH_ENTRY(unsigned short),
//            DISPATCH_ENTRY(int),
//            DISPATCH_ENTRY(unsigned int),
//            DISPATCH_ENTRY(long),
//            DISPATCH_ENTRY(unsigned long),
//            DISPATCH_ENTRY(long long),
//            DISPATCH_ENTRY(unsigned long long),
//            DISPATCH_ENTRY(float),
//            DISPATCH_ENTRY(double),
//            DISPATCH_ENTRY(long double),
//    };
//
//    auto it = dispatchMap.find(fieldType);
//
//    if (it != dispatchMap.end()) {
//        cerr<<"processing field "<<field_name<<endl;
//        return it->second(fieldPtr);
//    }
//
//    return false;
//}


