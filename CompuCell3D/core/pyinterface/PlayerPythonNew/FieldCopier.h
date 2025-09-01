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


        /**
         * Populates scalar Field3D with value of cell attribute (specified by the extractor function) in such a way
         * that all voxels occupied by a given cell will be initialized with the attribute value. For example
         * if we are populating cell_types we will get a Field where each voxel occupied by cell will get assigned
         * a value of the cell.type
         * @tparam T
         * @param fieldPtr
         * @param extractor - extracting function that fetches cell attribute
         * @return
         */
        template<typename T>
        bool fillCellAttributeValues(Field3D<T> *fieldPtr, std::function<T(CellG*)> extractor);

        /**
         * Copies voxel values from a *legacy* concentration field (float) to a
         * NumPy‑backed generic scalar field.
         *
         * @param legacy_field_name      – name registered in legacy map
         * @param destination_field_name – name of NumPy shared field to receive data
         * @return true on success, false on failure
         */
        bool copy_legacy_concentration_field(const std::string &source_field_name,
                                             const std::string &destination_field_name);


    private:
        std::tuple<std::type_index, void *> getFieldTypeAndPointer(const std::string &fieldName);

        typedef std::unordered_map<std::type_index, std::function<bool(void *)>> coreCopierFunctionMap_t;


    };
}

#endif
