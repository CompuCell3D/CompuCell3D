//
// Created by m on 3/22/25.
//

#ifndef COMPUCELL3D_FIELDCOPIER_H
#define COMPUCELL3D_FIELDCOPIER_H
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include "FieldExtractorDLLSpecifier.h"
#include <typeindex>


namespace CompuCell3D {

    //have to declare here all the classes that will be passed to this class from Python
    class Potts3D;

    class Simulator;
    class CellG;

    class Dim3D;

    class NeighborTracker;

    class ParallelUtilsOpenMP;

    template<typename T>
    class Field3D;

    class FIELDEXTRACTOR_EXPORT FieldCopier {
        Simulator *sim;
        Potts3D *potts;

    public:
        FieldCopier(Simulator *sim);
        ~FieldCopier()=default;
        bool copy_cell_attribute_field_values_to(const std::string& field_name, const std::string& attribute_name);
        std::vector<std::string> get_available_attributes();


        template<typename T>
        bool fillCellAttributeValues(Field3D<T> *fieldPtr, std::function<T(CellG*)> extractor);

    private:
        std::tuple<std::type_index, void *> getFieldTypeAndPointer(const std::string &fieldName);

        typedef std::unordered_map<std::type_index, std::function<bool(void *)>> coreCopierFunctionMap_t;

        template<typename T>
        bool fillCellTypeValues( Field3D<T> *fieldPtr);





    };
}

#endif
