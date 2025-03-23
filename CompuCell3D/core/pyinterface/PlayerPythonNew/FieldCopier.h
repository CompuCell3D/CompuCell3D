//
// Created by m on 3/22/25.
//

#ifndef COMPUCELL3D_FIELDCOPIER_H
#define COMPUCELL3D_FIELDCOPIER_H
#include <string>
#include <unordered_map>
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

    class FieldCopier {
        Simulator *sim;
        Potts3D *potts;

    public:
        FieldCopier(Simulator *sim);
        ~FieldCopier()=default;
        bool copy_cell_type_field_values_to(const std::string& field_name);
        bool copy_cell_attribute_field_values_to(const std::string& field_name, const std::string& attribute_name);

    private:
        std::tuple<std::type_index, void *> getFieldTypeAndPointer(const std::string &fieldName);
        template<typename T>
        bool fillCellAttributeValues(Field3D<T> *fieldPtr, std::function<T(CellG*)> extractor);

        typedef std::unordered_map<std::type_index, std::function<bool(void *)>> coreCopierFunctionMap_t;

        template<typename T>
        bool fillCellTypeValues( Field3D<T> *fieldPtr);

        void initializeCoreCopierFunctionMap();
        coreCopierFunctionMap_t coreCopierFunctionMap;



    };

} // CompuCell3D

#endif //COMPUCELL3D_FIELDCOPIER_H
