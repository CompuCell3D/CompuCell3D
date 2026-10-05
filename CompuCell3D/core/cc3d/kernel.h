#ifndef CC3D_KERNEL_H
#define CC3D_KERNEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CC3D_KERNEL_ABI_VERSION 30u

#if defined(_WIN32) || defined(__CYGWIN__)
#define CC3D_STEPPABLE_EXPORT __declspec(dllexport)
#else
#define CC3D_STEPPABLE_EXPORT
#endif

#ifdef __cplusplus
#include <string>
#include <initializer_list>
#include <utility>
#include <vector>
namespace cc3d {
namespace kernel {
class Cell;
class CellAttributes;
class SharedData;
class CellIterator;
class CellTypeRange;
class CellTypes;
class CellManipulation;
struct CellSentinel;
class ClusterIterator;
class ClusterRange;
class ClusterCellIterator;
class ClusterCellRange;
class CellNeighbor;
class CellNeighborIterator;
class CellNeighborRange;
struct CellNeighborSentinel;
class PixelIterator;
class PixelRange;
class EveryPixelRange;
struct PixelSentinel;
class OptionalCell;
class ScalarField;
class SharedScalarField;
class SharedVectorField;
class SBMLModel;
class AdhesionFlex;
class LengthConstraint;
class ClusterSurface;
class ClusterSurfaceTracker;
class Mitosis;
class OrientedGrowth;
class MomentOfInertia;
class FocalPointPlasticity;
class PolarizationVector;
class CellOrientation;
class Chemotaxis;
class ChemotaxisData;
class FieldSecretor;
class XmlElement;
struct Dim3;
}
}

extern "C" {
#endif

typedef void *CC3DCellIteratorHandle;
typedef void *CC3DClusterIteratorHandle;
typedef void *CC3DCellNeighborIteratorHandle;
typedef void *CC3DPixelIteratorHandle;
typedef void *CC3DFieldHandle;
typedef void *CC3DSharedFieldHandle;
typedef void *CC3DCellHandle;
typedef void *CC3DSBMLModelHandle;
typedef void *CC3DXMLHandle;
typedef void *CC3DFPPLinkIteratorHandle;
typedef void *CC3DFPPCellIteratorHandle;
typedef void *CC3DFPPLinkHandle;
typedef void *CC3DDictionaryIteratorHandle;

typedef enum CC3DDictionaryValueType {
    CC3D_DICTIONARY_VALUE_MISSING = 0,
    CC3D_DICTIONARY_VALUE_NONE = 1,
    CC3D_DICTIONARY_VALUE_BOOL = 2,
    CC3D_DICTIONARY_VALUE_INT = 3,
    CC3D_DICTIONARY_VALUE_DOUBLE = 4,
    CC3D_DICTIONARY_VALUE_STRING = 5,
    CC3D_DICTIONARY_VALUE_TUPLE = 6,
    CC3D_DICTIONARY_VALUE_LIST = 7,
    CC3D_DICTIONARY_VALUE_DICTIONARY = 8,
    CC3D_DICTIONARY_VALUE_OBJECT = 9
} CC3DDictionaryValueType;

typedef struct CC3DPythonDictionaryQueriesAPIV1 CC3DPythonDictionaryQueriesAPIV1;
typedef struct CC3DFieldDimV1 CC3DFieldDimV1;

typedef struct CC3DGeometryPointV1 {
    double x;
    double y;
    double z;
} CC3DGeometryPointV1;

typedef struct CC3DGeometryAPIV1 {
    void *userdata;
    CC3DGeometryPointV1 (*distance_vector_fn)(void *, CC3DGeometryPointV1, CC3DGeometryPointV1);
    double (*distance_fn)(void *, CC3DGeometryPointV1, CC3DGeometryPointV1);
    CC3DGeometryPointV1 (*invariant_distance_vector_fn)(void *, CC3DGeometryPointV1, CC3DGeometryPointV1);
    double (*invariant_distance_fn)(void *, CC3DGeometryPointV1, CC3DGeometryPointV1);
    CC3DGeometryPointV1 (*unconditional_invariant_distance_vector_fn)(void *, CC3DGeometryPointV1, CC3DGeometryPointV1);
    double (*vector_norm_fn)(void *, CC3DGeometryPointV1);
    CC3DGeometryPointV1 (*distance_vector_between_cells_fn)(void *, CC3DCellHandle, CC3DCellHandle, uint8_t invariant);
    double (*distance_between_cells_fn)(void *, CC3DCellHandle, CC3DCellHandle, uint8_t invariant);
    CC3DGeometryPointV1 (*hex_to_cartesian_fn)(void *, CC3DGeometryPointV1);
    CC3DGeometryPointV1 (*cartesian_to_hex_fn)(void *, CC3DGeometryPointV1);
    uint8_t (*is_in_bounds_fn)(void *, CC3DGeometryPointV1);
    CC3DFieldDimV1 (*field_dim_fn)(void *);
#ifdef __cplusplus
    inline CC3DGeometryPointV1 distanceVector(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const;
    inline double distance(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const;
    inline CC3DGeometryPointV1 invariantDistanceVector(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const;
    inline double invariantDistance(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const;
    inline CC3DGeometryPointV1 unconditionalInvariantDistanceVector(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const;
    inline double vectorNorm(CC3DGeometryPointV1 value) const;
    inline CC3DGeometryPointV1 distanceVectorBetweenCells(CC3DCellHandle a, CC3DCellHandle b, bool invariant = false) const;
    inline double distanceBetweenCells(CC3DCellHandle a, CC3DCellHandle b, bool invariant = false) const;
    inline CC3DGeometryPointV1 hexToCartesian(CC3DGeometryPointV1 value) const;
    inline CC3DGeometryPointV1 cartesianToHex(CC3DGeometryPointV1 value) const;
    inline bool isInBounds(CC3DGeometryPointV1 value) const;
    inline CC3DFieldDimV1 fieldDim() const;
#endif
} CC3DGeometryAPIV1;

typedef struct CC3DCellViewV1 {
    CC3DCellHandle handle;
    const long *id;
    const long *clusterId;
    const long *volume;
    const double *surface;
    const double *clusterSurface;
    unsigned char *type;
    unsigned char *subtype;
    float *targetVolume;
    float *lambdaVolume;
    float *targetSurface;
    float *angle;
    float *lambdaSurface;
    float *targetClusterSurface;
    float *lambdaClusterSurface;
    const double *xCM;
    const double *yCM;
    const double *zCM;
    const double *xCOM;
    const double *yCOM;
    const double *zCOM;
    const double *xCOMPrev;
    const double *yCOMPrev;
    const double *zCOMPrev;
    const double *iXX;
    const double *iXY;
    const double *iXZ;
    const double *iYY;
    const double *iYZ;
    const double *iZZ;
    const float *lX;
    const float *lY;
    const float *lZ;
    const float *ecc;
    float *lambdaVecX;
    float *lambdaVecY;
    float *lambdaVecZ;
    unsigned char *flag;
    float *averageConcentration;
    double *fluctAmpl;
    double *lambdaMotility;
    double *biasVecX;
    double *biasVecY;
    double *biasVecZ;
    bool *connectivityOn;
} CC3DCellViewV1;

typedef struct CC3DFPPLinkViewV1 {
    CC3DFPPLinkHandle handle;
    CC3DCellViewV1 initiator;
    CC3DCellViewV1 initiated;
    int anchorId;
    uint8_t valid;
} CC3DFPPLinkViewV1;

typedef struct CC3DCellTypesAPIV1 {
    void *userdata;
    uint8_t (*get_fn)(void *userdata, const char *type_name, unsigned char *type_id);
    size_t (*name_fn)(void *userdata, unsigned char type_id, char *buffer, size_t capacity);
#ifdef __cplusplus
    inline int get(const char *type_name) const;
    inline int operator[](const char *type_name) const;
    inline bool tryGet(const char *type_name, unsigned char &type_id) const;
    inline std::string name(unsigned char type_id) const;
    inline std::string name(const ::cc3d::kernel::Cell &cell) const;
    inline ::cc3d::kernel::CellTypes cellTypes() const;
#endif
} CC3DCellTypesAPIV1;

typedef struct CC3DPythonDictionaryAPIV1 {
    void *userdata;
    uint8_t (*contains_fn)(void *userdata, CC3DCellHandle cell, const char *key);
    void (*erase_fn)(void *userdata, CC3DCellHandle cell, const char *key);
    uint8_t (*set_bool_fn)(void *userdata, CC3DCellHandle cell, const char *key, uint8_t value);
    uint8_t (*set_int_fn)(void *userdata, CC3DCellHandle cell, const char *key, int64_t value);
    uint8_t (*set_double_fn)(void *userdata, CC3DCellHandle cell, const char *key, double value);
    uint8_t (*set_string_fn)(void *userdata, CC3DCellHandle cell, const char *key, const char *value);
    uint8_t (*set_tuple_fn)(void *userdata, CC3DCellHandle cell, const char *key,
                            const double *values, size_t count);
    uint8_t (*get_bool_fn)(void *userdata, CC3DCellHandle cell, const char *key, uint8_t *value);
    uint8_t (*get_int_fn)(void *userdata, CC3DCellHandle cell, const char *key, int64_t *value);
    uint8_t (*get_double_fn)(void *userdata, CC3DCellHandle cell, const char *key, double *value);
    size_t (*get_string_fn)(void *userdata, CC3DCellHandle cell, const char *key,
                            char *buffer, size_t capacity);
    size_t (*get_tuple_fn)(void *userdata, CC3DCellHandle cell, const char *key,
                           double *buffer, size_t capacity);
#ifdef __cplusplus
    inline ::cc3d::kernel::SharedData sharedData(
        const CC3DPythonDictionaryQueriesAPIV1 *queries = nullptr) const;
    inline ::cc3d::kernel::CellAttributes forCell(
        const ::cc3d::kernel::Cell &cell,
        const CC3DPythonDictionaryQueriesAPIV1 *queries = nullptr) const;
#endif
} CC3DPythonDictionaryAPIV1;

typedef struct CC3DXMLAPIV1 {
    void *userdata;
    CC3DXMLHandle (*get_element_fn)(void *userdata, const char *element_id);
    void (*destroy_element_fn)(void *userdata, CC3DXMLHandle element);
    uint8_t (*get_value_fn)(void *userdata, CC3DXMLHandle element,
                            const char *field, char *buffer, size_t capacity);
    uint8_t (*get_value_as_double_fn)(void *userdata, CC3DXMLHandle element,
                                      const char *field, double *value);
    uint8_t (*set_value_fn)(void *userdata, CC3DXMLHandle element,
                            const char *field, const char *value);
    uint8_t (*set_value_as_double_fn)(void *userdata, CC3DXMLHandle element,
                                      const char *field, double value);
#ifdef __cplusplus
    inline ::cc3d::kernel::XmlElement getElement(const char *element_id) const;
#endif
} CC3DXMLAPIV1;

typedef struct CC3DClusterCellsAPIV1 {
    void *userdata;
    CC3DClusterIteratorHandle (*clusters_begin_fn)(void *userdata);
    uint8_t (*clusters_valid_fn)(void *userdata, CC3DClusterIteratorHandle iterator);
    long (*clusters_get_id_fn)(void *userdata, CC3DClusterIteratorHandle iterator);
    void (*clusters_next_fn)(void *userdata, CC3DClusterIteratorHandle iterator);
    void (*clusters_destroy_fn)(void *userdata, CC3DClusterIteratorHandle iterator);
    CC3DCellIteratorHandle (*cells_begin_fn)(void *userdata, long cluster_id);
    uint8_t (*cells_valid_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    CC3DCellViewV1 (*cells_get_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    void (*cells_next_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    void (*cells_destroy_fn)(void *userdata, CC3DCellIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::ClusterRange clusters() const;
    inline ::cc3d::kernel::ClusterCellRange cells(long cluster_id) const;
#endif
} CC3DClusterCellsAPIV1;

typedef struct CC3DCellsAPIV1 {
    void *userdata;
    CC3DCellIteratorHandle (*begin_fn)(void *userdata);
    uint8_t (*valid_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    CC3DCellViewV1 (*get_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    void (*next_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    void (*destroy_fn)(void *userdata, CC3DCellIteratorHandle iterator);
    CC3DCellIteratorHandle (*by_type_begin_fn)(void *userdata, const unsigned char *type_ids, size_t type_count);
    CC3DCellViewV1 (*fetch_by_id_fn)(void *userdata, long cell_id);
    CC3DClusterCellsAPIV1 clusterApi;
#ifdef __cplusplus
    inline ::cc3d::kernel::CellIterator begin() const;
    inline ::cc3d::kernel::CellSentinel end() const;
    inline ::cc3d::kernel::CellTypeRange byType(unsigned char type_id) const;
    inline ::cc3d::kernel::CellTypeRange byType(const std::vector<unsigned char> &type_ids) const;
    inline ::cc3d::kernel::CellTypeRange byType(std::initializer_list<unsigned char> type_ids) const;
    inline ::cc3d::kernel::OptionalCell fetchCellById(long cell_id) const;
    inline ::cc3d::kernel::OptionalCell fetch_cell_by_id(long cell_id) const;
    inline ::cc3d::kernel::CellTypeRange cellListByType(unsigned char type_id) const;
    inline ::cc3d::kernel::CellTypeRange cell_list_by_type(unsigned char type_id) const;
    inline ::cc3d::kernel::CellTypeRange cellListByType(const std::vector<unsigned char> &type_ids) const;
    inline ::cc3d::kernel::CellTypeRange cell_list_by_type(const std::vector<unsigned char> &type_ids) const;
    inline ::cc3d::kernel::ClusterRange clusters() const;
    inline ::cc3d::kernel::ClusterCellRange clusterCells(long cluster_id) const;
#endif
} CC3DCellsAPIV1;

typedef struct CC3DCellNeighborViewV1 {
    CC3DCellViewV1 neighbor;
    int commonSurfaceArea;
} CC3DCellNeighborViewV1;

typedef struct CC3DCellNeighborsAPIV1 {
    void *userdata;
    CC3DCellNeighborIteratorHandle (*begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*valid_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
    CC3DCellNeighborViewV1 (*get_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
    void (*next_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
    void (*destroy_fn)(void *userdata, CC3DCellNeighborIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::CellNeighborRange forCell(const ::cc3d::kernel::Cell &cell) const;
#endif
} CC3DCellNeighborsAPIV1;

typedef struct CC3DPixelViewV1 {
    int x;
    int y;
    int z;
} CC3DPixelViewV1;

typedef struct CC3DCellPixelsAPIV1 {
    void *userdata;
    CC3DPixelIteratorHandle (*begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*valid_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
    CC3DPixelViewV1 (*get_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
    void (*next_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
    void (*destroy_fn)(void *userdata, CC3DPixelIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::PixelRange forCell(const ::cc3d::kernel::Cell &cell) const;
#endif
} CC3DCellPixelsAPIV1;

typedef struct CC3DFieldDimV1 {
    int x;
    int y;
    int z;
} CC3DFieldDimV1;

typedef struct CC3DOptionalCellViewV1 {
    uint8_t hasCell;
    CC3DCellViewV1 cell;
} CC3DOptionalCellViewV1;

typedef struct CC3DScalarFieldsAPIV1 {
    void *userdata;
    CC3DFieldHandle (*find_fn)(void *userdata, const char *name);
    CC3DFieldDimV1 (*dim_fn)(void *userdata, CC3DFieldHandle field);
    float (*get_fn)(void *userdata, CC3DFieldHandle field, int x, int y, int z);
    void (*set_fn)(void *userdata, CC3DFieldHandle field, int x, int y, int z, float value);
#ifdef __cplusplus
    inline ::cc3d::kernel::ScalarField operator[](const char *name) const;
#endif
} CC3DScalarFieldsAPIV1;

typedef enum CC3DFieldPrecisionV1 {
    CC3D_FIELD_FLOAT32 = 0,
    CC3D_FIELD_FLOAT64 = 1,
    CC3D_FIELD_INT16 = 2,
    CC3D_FIELD_UINT16 = 3,
    CC3D_FIELD_INT32 = 4,
    CC3D_FIELD_UINT32 = 5
} CC3DFieldPrecisionV1;

typedef struct CC3DSharedScalarFieldsAPIV1 {
    void *userdata;
    CC3DSharedFieldHandle (*find_fn)(void *userdata, const char *name);
    CC3DSharedFieldHandle (*create_fn)(void *userdata, const char *name, CC3DFieldPrecisionV1 precision, int padding);
    CC3DFieldDimV1 (*dim_fn)(void *userdata, CC3DSharedFieldHandle field, uint8_t raw);
    double (*get_fn)(void *userdata, CC3DSharedFieldHandle field, int x, int y, int z, uint8_t raw);
    uint8_t (*set_fn)(void *userdata, CC3DSharedFieldHandle field, int x, int y, int z, double value, uint8_t raw);
#ifdef __cplusplus
    inline ::cc3d::kernel::SharedScalarField scalar(const char *name) const;
    inline ::cc3d::kernel::SharedScalarField createScalar(const char *name, CC3DFieldPrecisionV1 precision = CC3D_FIELD_FLOAT32, int padding = 0) const;
#endif
} CC3DSharedScalarFieldsAPIV1;

typedef struct CC3DSharedVectorFieldsAPIV1 {
    void *userdata;
    CC3DSharedFieldHandle (*find_fn)(void *userdata, const char *name);
    CC3DSharedFieldHandle (*create_fn)(void *userdata, const char *name, CC3DFieldPrecisionV1 precision, int padding);
    CC3DFieldDimV1 (*dim_fn)(void *userdata, CC3DSharedFieldHandle field, uint8_t raw);
    uint8_t (*get_fn)(void *userdata, CC3DSharedFieldHandle field, int x, int y, int z, double *vx, double *vy, double *vz, uint8_t raw);
    uint8_t (*set_fn)(void *userdata, CC3DSharedFieldHandle field, int x, int y, int z, double vx, double vy, double vz, uint8_t raw);
#ifdef __cplusplus
    inline ::cc3d::kernel::SharedVectorField vector(const char *name) const;
    inline ::cc3d::kernel::SharedVectorField createVector(const char *name, CC3DFieldPrecisionV1 precision = CC3D_FIELD_FLOAT32, int padding = 0) const;
#endif
} CC3DSharedVectorFieldsAPIV1;

typedef struct CC3DCellFieldAPIV1 {
    void *userdata;
    CC3DFieldDimV1 (*dim_fn)(void *userdata);
    CC3DOptionalCellViewV1 (*get_fn)(void *userdata, int x, int y, int z);
    void (*assign_fn)(void *userdata, int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, CC3DCellHandle cell);
#ifdef __cplusplus
    inline ::cc3d::kernel::Dim3 dim() const;
    inline ::cc3d::kernel::OptionalCell operator()(int x, int y, int z) const;
    inline ::cc3d::kernel::EveryPixelRange everyPixel(int step_x = 1, int step_y = 1, int step_z = 1) const;
    inline ::cc3d::kernel::EveryPixelRange everyPixelWithSteps(int step_x, int step_y, int step_z) const;
    inline void assign(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, const ::cc3d::kernel::Cell &cell) const;
#endif
} CC3DCellFieldAPIV1;

typedef struct CC3DCellManipulationAPIV1 {
    void *userdata;
    uint8_t (*delete_cell_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*move_cell_fn)(void *userdata, CC3DCellHandle cell, int dx, int dy, int dz);
    uint8_t (*merge_cells_fn)(void *userdata, CC3DCellHandle source, CC3DCellHandle destination);
#ifdef __cplusplus
    inline ::cc3d::kernel::CellManipulation cellManipulation() const;
#endif
} CC3DCellManipulationAPIV1;

typedef struct CC3DSBMLAPIV1 {
    void *userdata;
    CC3DSBMLModelHandle (*find_model_fn)(void *userdata, CC3DCellHandle cell, const char *model_name);
    void (*destroy_model_fn)(void *userdata, CC3DSBMLModelHandle model);
    uint8_t (*has_value_fn)(void *userdata, CC3DSBMLModelHandle model, const char *value_name);
    uint8_t (*get_value_fn)(void *userdata, CC3DSBMLModelHandle model, const char *value_name, double *value);
    uint8_t (*set_value_fn)(void *userdata, CC3DSBMLModelHandle model, const char *value_name, double value);
    uint8_t (*timestep_fn)(void *userdata, CC3DSBMLModelHandle model);
#ifdef __cplusplus
    inline ::cc3d::kernel::SBMLModel find(const ::cc3d::kernel::Cell &cell, const char *model_name) const;
#endif
} CC3DSBMLAPIV1;
typedef struct CC3DCellFactoryAPIV1 {
    void *userdata;
    CC3DCellViewV1 (*create_fn)(void *userdata, unsigned char cell_type);
#ifdef __cplusplus
    inline ::cc3d::kernel::Cell create(unsigned char cell_type) const;
#endif
} CC3DCellFactoryAPIV1;

typedef struct CC3DAdhesionFlexAPIV1 {
    void *userdata;
    size_t (*molecule_count_fn)(void *userdata);
    const char *(*molecule_name_fn)(void *userdata, size_t index);
    float (*get_density_fn)(void *userdata, CC3DCellHandle cell, const char *molecule_name);
    float (*get_density_by_index_fn)(void *userdata, CC3DCellHandle cell, size_t index);
    void (*set_density_fn)(void *userdata, CC3DCellHandle cell, const char *molecule_name, float density);
    void (*set_density_by_index_fn)(void *userdata, CC3DCellHandle cell, size_t index, float density);
    uint8_t (*set_density_vector_fn)(void *userdata, CC3DCellHandle cell, const float *densities, size_t count);
    void (*assign_density_vector_fn)(void *userdata, CC3DCellHandle cell, const float *densities, size_t count);
    float (*get_medium_density_fn)(void *userdata, const char *molecule_name);
    float (*get_medium_density_by_index_fn)(void *userdata, size_t index);
    void (*set_medium_density_fn)(void *userdata, const char *molecule_name, float density);
    void (*set_medium_density_by_index_fn)(void *userdata, size_t index, float density);
    uint8_t (*set_medium_density_vector_fn)(void *userdata, const float *densities, size_t count);
    void (*assign_medium_density_vector_fn)(void *userdata, const float *densities, size_t count);
#ifdef __cplusplus
    inline ::cc3d::kernel::AdhesionFlex adhesionFlex() const;
#endif
} CC3DAdhesionFlexAPIV1;

typedef struct CC3DLengthConstraintAPIV1 {
    void *userdata;
    void (*set_fn)(void *userdata, CC3DCellHandle cell, double lambda_length,
                   double target_length, double minor_target_length);
    double (*get_lambda_fn)(void *userdata, CC3DCellHandle cell);
    double (*get_target_fn)(void *userdata, CC3DCellHandle cell);
    double (*get_minor_target_fn)(void *userdata, CC3DCellHandle cell);
#ifdef __cplusplus
    inline ::cc3d::kernel::LengthConstraint lengthConstraint() const;
#endif
} CC3DLengthConstraintAPIV1;

typedef struct CC3DClusterSurfaceAPIV1 {
    void *userdata;
    void (*set_fn)(void *userdata, CC3DCellHandle cell, float target_cluster_surface,
                   float lambda_cluster_surface);
    void (*get_fn)(void *userdata, CC3DCellHandle cell, float *target_cluster_surface,
                   float *lambda_cluster_surface);
#ifdef __cplusplus
    inline ::cc3d::kernel::ClusterSurface clusterSurface() const;
#endif
} CC3DClusterSurfaceAPIV1;

typedef struct CC3DClusterSurfaceTrackerAPIV1 {
    void *userdata;
    void (*update_cluster_surface_fn)(void *userdata, long cluster_id);
    uint8_t (*reassign_cluster_id_fn)(void *userdata, CC3DCellHandle cell, long cluster_id);
#ifdef __cplusplus
    inline ::cc3d::kernel::ClusterSurfaceTracker clusterSurfaceTracker() const;
#endif
} CC3DClusterSurfaceTrackerAPIV1;

typedef struct CC3DMitosisAPIV1 {
    void *userdata;
    void (*set_parent_child_position_fn)(void *, int);
    int (*get_parent_child_position_fn)(void *);
    uint8_t (*divide_random_fn)(void *, CC3DCellHandle);
    uint8_t (*divide_orientation_fn)(void *, CC3DCellHandle, double, double, double);
    uint8_t (*divide_major_axis_fn)(void *, CC3DCellHandle);
    uint8_t (*divide_minor_axis_fn)(void *, CC3DCellHandle);
    uint8_t (*divide_cluster_random_fn)(void *, long);
    uint8_t (*divide_cluster_orientation_fn)(void *, long, double, double, double);
    uint8_t (*divide_cluster_major_axis_fn)(void *, long);
    uint8_t (*divide_cluster_minor_axis_fn)(void *, long);
    CC3DCellViewV1 (*parent_cell_fn)(void *);
    CC3DCellViewV1 (*child_cell_fn)(void *);
    void (*clone_attributes_fn)(void *, CC3DCellHandle, CC3DCellHandle);
#ifdef __cplusplus
    inline ::cc3d::kernel::Mitosis mitosis() const;
#endif
} CC3DMitosisAPIV1;

typedef struct CC3DOrientedGrowthAPIV1 {
    void *userdata;
    void (*set_constraint_width_fn)(void *, CC3DCellHandle, float);
    void (*set_elongation_axis_fn)(void *, CC3DCellHandle, float, float);
    void (*set_elongation_enabled_fn)(void *, CC3DCellHandle, uint8_t);
    float (*get_elongation_axis_x_fn)(void *, CC3DCellHandle);
    float (*get_elongation_axis_y_fn)(void *, CC3DCellHandle);
    uint8_t (*get_elongation_enabled_fn)(void *, CC3DCellHandle);
#ifdef __cplusplus
    inline ::cc3d::kernel::OrientedGrowth orientedGrowth() const;
#endif
} CC3DOrientedGrowthAPIV1;

typedef struct CC3DMomentOfInertiaAPIV1 {
    void *userdata;
    size_t (*get_semiaxes_fn)(void *, CC3DCellHandle, double *, size_t);
    size_t (*get_semiaxes_xy_fn)(void *, CC3DCellHandle, double *, size_t);
    size_t (*get_semiaxes_xz_fn)(void *, CC3DCellHandle, double *, size_t);
    size_t (*get_semiaxes_yz_fn)(void *, CC3DCellHandle, double *, size_t);
    size_t (*get_semiaxes_3d_fn)(void *, CC3DCellHandle, double *, size_t);
#ifdef __cplusplus
    inline ::cc3d::kernel::MomentOfInertia momentOfInertia() const;
#endif
} CC3DMomentOfInertiaAPIV1;

typedef struct CC3DFocalPointPlasticityAPIV1 {
    void *userdata;
    void (*create_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2,
                           double lambda, double target_distance, double max_distance);
    void (*create_internal_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2,
                                    double lambda, double target_distance, double max_distance);
    void (*delete_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2);
    void (*delete_internal_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2);
    void (*set_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2,
                        double lambda, double target_distance, double max_distance);
    double (*get_lambda_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2);
    double (*get_target_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2);
    int (*create_anchor_fn)(void *userdata, CC3DCellHandle cell, double lambda,
                            double target_distance, double max_distance, float x, float y, float z);
    void (*delete_anchor_fn)(void *userdata, CC3DCellHandle cell, int anchor_id);
    void (*set_anchor_fn)(void *userdata, CC3DCellHandle cell, int anchor_id, double lambda,
                          double target_distance, double max_distance, float x, float y, float z);
#ifdef __cplusplus
    inline ::cc3d::kernel::FocalPointPlasticity focalPointPlasticity() const;
#endif
} CC3DFocalPointPlasticityAPIV1;

// Query operations are kept in a separate table because the base FPP table is
// embedded in the middle of CC3DKernelContext. Appending fields to that base
// table would move every following context member and corrupt older modules.
typedef struct CC3DFocalPointPlasticityQueriesAPIV1 {
    void *userdata;
    CC3DFPPLinkIteratorHandle (*links_begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*links_valid_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    CC3DFPPLinkViewV1 (*links_get_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    void (*links_next_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    void (*links_destroy_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    CC3DFPPLinkIteratorHandle (*internal_links_begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*internal_links_valid_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    CC3DFPPLinkViewV1 (*internal_links_get_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    void (*internal_links_next_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    void (*internal_links_destroy_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    CC3DFPPLinkIteratorHandle (*anchors_begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*anchors_valid_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    CC3DFPPLinkViewV1 (*anchors_get_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    void (*anchors_next_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    void (*anchors_destroy_fn)(void *userdata, CC3DFPPLinkIteratorHandle iterator);
    CC3DFPPLinkViewV1 (*find_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2);
    CC3DFPPLinkViewV1 (*find_internal_link_fn)(void *userdata, CC3DCellHandle cell1, CC3DCellHandle cell2);
    size_t (*link_count_fn)(void *userdata, CC3DCellHandle cell);
    size_t (*internal_link_count_fn)(void *userdata, CC3DCellHandle cell);
    size_t (*anchor_count_fn)(void *userdata, CC3DCellHandle cell);
    CC3DFPPCellIteratorHandle (*linked_cells_begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*linked_cells_valid_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    CC3DCellViewV1 (*linked_cells_get_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    void (*linked_cells_next_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    void (*linked_cells_destroy_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    CC3DFPPCellIteratorHandle (*internal_linked_cells_begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*internal_linked_cells_valid_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    CC3DCellViewV1 (*internal_linked_cells_get_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    void (*internal_linked_cells_next_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
    void (*internal_linked_cells_destroy_fn)(void *userdata, CC3DFPPCellIteratorHandle iterator);
#ifdef __cplusplus
    inline ::cc3d::kernel::FocalPointPlasticity focalPointPlasticity(
        const CC3DFocalPointPlasticityAPIV1 *base) const;
#endif
} CC3DFocalPointPlasticityQueriesAPIV1;

// Dictionary extensions live in a separate table because both dictionary
// tables are embedded in the middle of CC3DKernelContext. Do not append these
// operations to CC3DPythonDictionaryAPIV1; doing so would shift all following
// context members and break separately compiled modules.
typedef struct CC3DPythonDictionaryQueriesAPIV1 {
    void *userdata;
    CC3DDictionaryIteratorHandle (*keys_begin_fn)(void *userdata, CC3DCellHandle cell);
    uint8_t (*keys_valid_fn)(void *userdata, CC3DDictionaryIteratorHandle iterator);
    size_t (*keys_get_fn)(void *userdata, CC3DDictionaryIteratorHandle iterator,
                          char *buffer, size_t capacity);
    void (*keys_next_fn)(void *userdata, CC3DDictionaryIteratorHandle iterator);
    void (*keys_destroy_fn)(void *userdata, CC3DDictionaryIteratorHandle iterator);
    CC3DDictionaryValueType (*get_type_fn)(void *userdata, CC3DCellHandle cell, const char *key);
    size_t (*get_size_fn)(void *userdata, CC3DCellHandle cell, const char *key);
    uint8_t (*set_none_fn)(void *userdata, CC3DCellHandle cell, const char *key);
    uint8_t (*set_list_fn)(void *userdata, CC3DCellHandle cell, const char *key,
                           const double *values, size_t count);
    size_t (*get_list_fn)(void *userdata, CC3DCellHandle cell, const char *key,
                          double *buffer, size_t capacity);
} CC3DPythonDictionaryQueriesAPIV1;

typedef struct CC3DPolarizationVectorAPIV1 {
    void *userdata;
    void (*set_fn)(void *, CC3DCellHandle, float, float, float);
    uint8_t (*get_fn)(void *, CC3DCellHandle, float *, float *, float *);
#ifdef __cplusplus
    inline ::cc3d::kernel::PolarizationVector polarizationVector() const;
#endif
} CC3DPolarizationVectorAPIV1;

typedef struct CC3DCellOrientationAPIV1 {
    void *userdata;
    void (*set_lambda_fn)(void *, CC3DCellHandle, double);
    double (*get_lambda_fn)(void *, CC3DCellHandle);
#ifdef __cplusplus
    inline ::cc3d::kernel::CellOrientation cellOrientation() const;
#endif
} CC3DCellOrientationAPIV1;

typedef struct CC3DChemotaxisDataAPIV1 {
    void *userdata;
    float (*get_lambda_fn)(void *);
    void (*set_lambda_fn)(void *, float);
    float (*get_saturation_fn)(void *);
    void (*set_saturation_fn)(void *, float);
    const char *(*get_formula_name_fn)(void *);
    uint8_t (*set_formula_name_fn)(void *, const char *);
    size_t (*get_toward_count_fn)(void *);
    int (*get_toward_type_fn)(void *, size_t);
    void (*assign_toward_types_fn)(void *, const int *, size_t);
} CC3DChemotaxisDataAPIV1;

typedef struct CC3DChemotaxisAPIV1 {
    void *userdata;
    size_t (*field_count_fn)(void *, CC3DCellHandle);
    const char *(*field_name_fn)(void *, CC3DCellHandle, size_t);
    CC3DChemotaxisDataAPIV1 (*add_data_fn)(void *, CC3DCellHandle, const char *);
    CC3DChemotaxisDataAPIV1 (*get_data_fn)(void *, CC3DCellHandle, const char *);
#ifdef __cplusplus
    inline ::cc3d::kernel::Chemotaxis chemotaxis() const;
#endif
} CC3DChemotaxisAPIV1;

#ifndef CC3D_FIELD_SECRETOR_RESULT_V1_DEFINED
#define CC3D_FIELD_SECRETOR_RESULT_V1_DEFINED
typedef struct CC3DFieldSecretorResultV1 {
    uint8_t success;
    float total_amount;
} CC3DFieldSecretorResultV1;
#endif

typedef struct CC3DSecretionAPIV1 {
    void *userdata;
    uint8_t (*secrete_inside_fn)(void *, CC3DCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_inside_total_fn)(void *, CC3DCellHandle, const char *, float);
    uint8_t (*secrete_inside_constant_fn)(void *, CC3DCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_inside_constant_total_fn)(void *, CC3DCellHandle, const char *, float);
    uint8_t (*secrete_inside_boundary_fn)(void *, CC3DCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_inside_boundary_total_fn)(void *, CC3DCellHandle, const char *, float);
    uint8_t (*secrete_inside_boundary_contact_fn)(void *, CC3DCellHandle, const char *, float, const unsigned char *, size_t);
    CC3DFieldSecretorResultV1 (*secrete_inside_boundary_contact_total_fn)(void *, CC3DCellHandle, const char *, float, const unsigned char *, size_t);
    uint8_t (*secrete_outside_boundary_fn)(void *, CC3DCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_outside_boundary_total_fn)(void *, CC3DCellHandle, const char *, float);
    uint8_t (*secrete_outside_boundary_contact_fn)(void *, CC3DCellHandle, const char *, float, const unsigned char *, size_t);
    CC3DFieldSecretorResultV1 (*secrete_outside_boundary_contact_total_fn)(void *, CC3DCellHandle, const char *, float, const unsigned char *, size_t);
    uint8_t (*secrete_com_fn)(void *, CC3DCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_com_total_fn)(void *, CC3DCellHandle, const char *, float);
    uint8_t (*uptake_inside_fn)(void *, CC3DCellHandle, const char *, float, float);
    CC3DFieldSecretorResultV1 (*uptake_inside_total_fn)(void *, CC3DCellHandle, const char *, float, float);
    uint8_t (*uptake_boundary_fn)(void *, CC3DCellHandle, const char *, float, float);
    CC3DFieldSecretorResultV1 (*uptake_boundary_total_fn)(void *, CC3DCellHandle, const char *, float, float);
    uint8_t (*uptake_outside_boundary_fn)(void *, CC3DCellHandle, const char *, float, float);
    CC3DFieldSecretorResultV1 (*uptake_outside_boundary_total_fn)(void *, CC3DCellHandle, const char *, float, float);
    float (*amount_seen_fn)(void *, CC3DCellHandle, const char *);
    float (*field_integral_fn)(void *, const char *);
    void (*secrete_single_field_fn)(void *, size_t);
    void (*secrete_contact_single_field_fn)(void *, size_t);
    void (*secrete_constant_single_field_fn)(void *, size_t);
#ifdef __cplusplus
    inline ::cc3d::kernel::FieldSecretor fieldSecretor() const;
    inline ::cc3d::kernel::FieldSecretor fieldSecretor(const char *field) const;
#endif
} CC3DSecretionAPIV1;

typedef struct CC3DKernelContext {
    // ABI layout rule: this context is shared with separately compiled DLLs.
    // Never insert fields, reorder fields, or grow a struct already embedded
    // here. This includes nested tables such as CC3DFocalPointPlasticityAPIV1:
    // growing one of them shifts every member that follows it. Existing
    // members keep their byte offsets forever. New APIs must be represented by
    // a new table appended at the end of this struct, followed by an ABI
    // version bump and a full rebuild of the host and all compiled modules.
    uint32_t abiVersion;
    uint64_t mcs;
    CC3DCellTypesAPIV1 cellTypes;
    CC3DCellsAPIV1 cells;
    CC3DCellNeighborsAPIV1 neighbors;
    CC3DCellPixelsAPIV1 pixels;
    CC3DCellPixelsAPIV1 boundaryPixels;
    CC3DScalarFieldsAPIV1 scalarFields;
    CC3DCellFieldAPIV1 cellField;
    CC3DCellManipulationAPIV1 cellManipulation;
    CC3DCellFactoryAPIV1 cellFactory;
    CC3DSBMLAPIV1 sbml;
    CC3DAdhesionFlexAPIV1 adhesionFlex;
    CC3DLengthConstraintAPIV1 lengthConstraint;
    CC3DClusterSurfaceAPIV1 clusterSurface;
    CC3DClusterSurfaceTrackerAPIV1 clusterSurfaceTracker;
    CC3DMitosisAPIV1 mitosis;
    CC3DOrientedGrowthAPIV1 orientedGrowth;
    CC3DMomentOfInertiaAPIV1 momentOfInertia;
    CC3DFocalPointPlasticityAPIV1 focalPointPlasticity;
    CC3DPolarizationVectorAPIV1 polarizationVector;
    CC3DCellOrientationAPIV1 cellOrientation;
    CC3DChemotaxisAPIV1 chemotaxis;
    CC3DSecretionAPIV1 secretion;
    CC3DPythonDictionaryAPIV1 cellAttributes;
    CC3DPythonDictionaryAPIV1 sharedData;
    CC3DXMLAPIV1 xml;
    // Append newly introduced APIs so existing context member offsets remain stable.
    CC3DSharedScalarFieldsAPIV1 sharedScalarFields;
    CC3DSharedVectorFieldsAPIV1 sharedVectorFields;
    CC3DFocalPointPlasticityQueriesAPIV1 focalPointPlasticityQueries;
    CC3DPythonDictionaryQueriesAPIV1 dictionaryQueries;
    CC3DGeometryAPIV1 geometry;
#ifdef __cplusplus
    inline ::cc3d::kernel::Cell newCell(unsigned char cell_type) const;
#endif
} CC3DKernelContext;

typedef struct CC3DSteppableV1 {
    uint32_t abiVersion;
    void *(*create_fn)();
    void (*start_fn)(CC3DKernelContext *ctx, void *state);
    void (*step_fn)(CC3DKernelContext *ctx, void *state);
    void (*finish_fn)(CC3DKernelContext *ctx, void *state);
    void (*destroy_fn)(void *state);
    // Optional initialization callback. It runs after the simulator context
    // is attached and before start_fn, allowing simulator-owned fields and
    // other resources to be created before the simulation starts.
    void (*init_fn)(CC3DKernelContext *ctx, void *state);
} CC3DSteppableV1;


typedef const CC3DSteppableV1 *(*CC3DGetSteppableV1Fn)();

#ifdef __cplusplus
}

namespace cc3d {
namespace kernel {

struct CellSentinel {};

struct CellNeighborSentinel {};

struct PixelSentinel {};

struct Dim3 {
    int x;
    int y;
    int z;
};

template<typename T>
class ReadOnlyProperty {
public:
    ReadOnlyProperty() : ptr_(nullptr) {}
    explicit ReadOnlyProperty(const T *ptr) : ptr_(ptr) {}

    operator T() const { return ptr_ ? *ptr_ : T(); }

private:
    const T *ptr_;
};

template<typename T>
class Property {
public:
    Property() : ptr_(nullptr) {}
    explicit Property(T *ptr) : ptr_(ptr) {}

    operator T() const { return ptr_ ? *ptr_ : T(); }

    Property &operator=(T value) {
        if (ptr_) {
            *ptr_ = value;
        }
        return *this;
    }

    Property &operator+=(T value) {
        if (ptr_) {
            *ptr_ += value;
        }
        return *this;
    }

    Property &operator-=(T value) {
        if (ptr_) {
            *ptr_ -= value;
        }
        return *this;
    }

private:
    T *ptr_;
};

class Cell {
public:
    explicit Cell(const CC3DCellViewV1 &view)
        : handle_(view.handle),
          id(view.id),
          clusterId(view.clusterId),
          volume(view.volume),
          surface(view.surface),
          clusterSurface(view.clusterSurface),
          type(view.type),
          subtype(view.subtype),
          targetVolume(view.targetVolume),
          lambdaVolume(view.lambdaVolume),
          targetSurface(view.targetSurface),
          angle(view.angle),
          lambdaSurface(view.lambdaSurface),
          targetClusterSurface(view.targetClusterSurface),
          lambdaClusterSurface(view.lambdaClusterSurface),
          xCM(view.xCM),
          yCM(view.yCM),
          zCM(view.zCM),
          xCOM(view.xCOM),
          yCOM(view.yCOM),
          zCOM(view.zCOM),
          xCOMPrev(view.xCOMPrev),
          yCOMPrev(view.yCOMPrev),
          zCOMPrev(view.zCOMPrev),
          iXX(view.iXX),
          iXY(view.iXY),
          iXZ(view.iXZ),
          iYY(view.iYY),
          iYZ(view.iYZ),
          iZZ(view.iZZ),
          lX(view.lX),
          lY(view.lY),
          lZ(view.lZ),
          ecc(view.ecc),
          lambdaVecX(view.lambdaVecX),
          lambdaVecY(view.lambdaVecY),
          lambdaVecZ(view.lambdaVecZ),
          flag(view.flag),
          averageConcentration(view.averageConcentration),
          fluctAmpl(view.fluctAmpl),
          lambdaMotility(view.lambdaMotility),
          biasVecX(view.biasVecX),
          biasVecY(view.biasVecY),
          biasVecZ(view.biasVecZ),
          connectivityOn(view.connectivityOn) {}

    CC3DCellHandle handle() const { return handle_; }

    ReadOnlyProperty<long> id;
    ReadOnlyProperty<long> clusterId;
    ReadOnlyProperty<long> volume;
    ReadOnlyProperty<double> surface;
    ReadOnlyProperty<double> clusterSurface;
    Property<unsigned char> type;
    Property<unsigned char> subtype;
    Property<float> targetVolume;
    Property<float> lambdaVolume;
    Property<float> targetSurface;
    Property<float> angle;
    Property<float> lambdaSurface;
    Property<float> targetClusterSurface;
    Property<float> lambdaClusterSurface;
    ReadOnlyProperty<double> xCM;
    ReadOnlyProperty<double> yCM;
    ReadOnlyProperty<double> zCM;
    ReadOnlyProperty<double> xCOM;
    ReadOnlyProperty<double> yCOM;
    ReadOnlyProperty<double> zCOM;
    ReadOnlyProperty<double> xCOMPrev;
    ReadOnlyProperty<double> yCOMPrev;
    ReadOnlyProperty<double> zCOMPrev;
    ReadOnlyProperty<double> iXX;
    ReadOnlyProperty<double> iXY;
    ReadOnlyProperty<double> iXZ;
    ReadOnlyProperty<double> iYY;
    ReadOnlyProperty<double> iYZ;
    ReadOnlyProperty<double> iZZ;
    ReadOnlyProperty<float> lX;
    ReadOnlyProperty<float> lY;
    ReadOnlyProperty<float> lZ;
    ReadOnlyProperty<float> ecc;
    Property<float> lambdaVecX;
    Property<float> lambdaVecY;
    Property<float> lambdaVecZ;
    Property<unsigned char> flag;
    Property<float> averageConcentration;
    Property<double> fluctAmpl;
    Property<double> lambdaMotility;
    Property<double> biasVecX;
    Property<double> biasVecY;
    Property<double> biasVecZ;
    Property<bool> connectivityOn;

private:
    CC3DCellHandle handle_;
};

class CellTypes {
public:
    CellTypes() : api_(nullptr) {}
    explicit CellTypes(const CC3DCellTypesAPIV1 *api) : api_(api) {}

    // Returns -1 when the name is unknown or the CellType plugin is unavailable.
    int get(const char *type_name) const {
        unsigned char type_id = 0;
        return tryGet(type_name, type_id) ? static_cast<int>(type_id) : -1;
    }

    bool tryGet(const char *type_name, unsigned char &type_id) const {
        return api_ && api_->get_fn && type_name &&
               api_->get_fn(api_->userdata, type_name, &type_id) != 0;
    }

    std::string name(unsigned char type_id) const {
        if (!api_ || !api_->name_fn) return {};
        const size_t size = api_->name_fn(api_->userdata, type_id, nullptr, 0);
        std::string result(size, '\0');
        if (size) api_->name_fn(api_->userdata, type_id, &result[0], size + 1);
        return result;
    }

    std::string name(const Cell &cell) const {
        return name(static_cast<unsigned char>(cell.type));
    }

private:
    const CC3DCellTypesAPIV1 *api_;
};

class CellManipulation {
public:
    CellManipulation() : api_(nullptr) {}
    explicit CellManipulation(const CC3DCellManipulationAPIV1 *api) : api_(api) {}

    bool deleteCell(const Cell &cell) const {
        return api_ && api_->delete_cell_fn &&
               api_->delete_cell_fn(api_->userdata, cell.handle()) != 0;
    }

    bool moveCell(const Cell &cell, int dx, int dy, int dz) const {
        return api_ && api_->move_cell_fn &&
               api_->move_cell_fn(api_->userdata, cell.handle(), dx, dy, dz) != 0;
    }

    bool mergeCells(const Cell &source, const Cell &destination) const {
        return api_ && api_->merge_cells_fn &&
               api_->merge_cells_fn(api_->userdata, source.handle(), destination.handle()) != 0;
    }

private:
    const CC3DCellManipulationAPIV1 *api_;
};

class PythonDictionary {
public:
    PythonDictionary() : api_(nullptr), queries_(nullptr), cell_(nullptr) {}
    PythonDictionary(const CC3DPythonDictionaryAPIV1 *api, CC3DCellHandle cell,
                     const CC3DPythonDictionaryQueriesAPIV1 *queries = nullptr)
        : api_(api), queries_(queries), cell_(cell) {}

    bool contains(const char *key) const {
        return api_ && api_->contains_fn && key && api_->contains_fn(api_->userdata, cell_, key) != 0;
    }
    void erase(const char *key) const {
        if (api_ && api_->erase_fn && key) api_->erase_fn(api_->userdata, cell_, key);
    }
    bool setBool(const char *key, bool value) const {
        return api_ && api_->set_bool_fn && key && api_->set_bool_fn(api_->userdata, cell_, key, value ? 1u : 0u) != 0;
    }
    bool setInt(const char *key, int64_t value) const {
        return api_ && api_->set_int_fn && key && api_->set_int_fn(api_->userdata, cell_, key, value) != 0;
    }
    bool setDouble(const char *key, double value) const {
        return api_ && api_->set_double_fn && key && api_->set_double_fn(api_->userdata, cell_, key, value) != 0;
    }
    bool setString(const char *key, const char *value) const {
        return api_ && api_->set_string_fn && key && value && api_->set_string_fn(api_->userdata, cell_, key, value) != 0;
    }
    bool setTuple(const char *key, const std::vector<double> &values) const {
        return api_ && api_->set_tuple_fn && key && api_->set_tuple_fn(api_->userdata, cell_, key, values.data(), values.size()) != 0;
    }
    bool setNone(const char *key) const {
        return queries_ && queries_->set_none_fn && key &&
               queries_->set_none_fn(queries_->userdata, cell_, key) != 0;
    }
    bool setList(const char *key, const std::vector<double> &values) const {
        return queries_ && queries_->set_list_fn && key &&
               queries_->set_list_fn(queries_->userdata, cell_, key, values.data(), values.size()) != 0;
    }
    CC3DDictionaryValueType type(const char *key) const {
        return queries_ && queries_->get_type_fn && key
            ? queries_->get_type_fn(queries_->userdata, cell_, key)
            : CC3D_DICTIONARY_VALUE_MISSING;
    }
    size_t size(const char *key) const {
        return queries_ && queries_->get_size_fn && key
            ? queries_->get_size_fn(queries_->userdata, cell_, key)
            : 0u;
    }
    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        if (!queries_ || !queries_->keys_begin_fn || !queries_->keys_valid_fn ||
            !queries_->keys_get_fn || !queries_->keys_next_fn ||
            !queries_->keys_destroy_fn) return result;
        CC3DDictionaryIteratorHandle iterator =
            queries_->keys_begin_fn(queries_->userdata, cell_);
        while (iterator && queries_->keys_valid_fn(queries_->userdata, iterator)) {
            const size_t length = queries_->keys_get_fn(
                queries_->userdata, iterator, nullptr, 0u);
            std::string key(length, '\0');
            if (length) {
                queries_->keys_get_fn(queries_->userdata, iterator, &key[0], length + 1u);
            }
            result.push_back(std::move(key));
            queries_->keys_next_fn(queries_->userdata, iterator);
        }
        if (iterator) queries_->keys_destroy_fn(queries_->userdata, iterator);
        return result;
    }
    bool getBool(const char *key, bool &value) const {
        uint8_t result = 0;
        if (!api_ || !api_->get_bool_fn || !key || !api_->get_bool_fn(api_->userdata, cell_, key, &result)) return false;
        value = result != 0;
        return true;
    }
    bool getInt(const char *key, int64_t &value) const {
        return api_ && api_->get_int_fn && key && api_->get_int_fn(api_->userdata, cell_, key, &value) != 0;
    }
    bool getDouble(const char *key, double &value) const {
        return api_ && api_->get_double_fn && key && api_->get_double_fn(api_->userdata, cell_, key, &value) != 0;
    }
    std::string getString(const char *key) const {
        if (!api_ || !api_->get_string_fn || !key) return {};
        const size_t size = api_->get_string_fn(api_->userdata, cell_, key, nullptr, 0);
        std::string result(size, '\0');
        if (size) api_->get_string_fn(api_->userdata, cell_, key, &result[0], size + 1);
        return result;
    }
    std::vector<double> getTuple(const char *key) const {
        if (!api_ || !api_->get_tuple_fn || !key) return {};
        const size_t size = api_->get_tuple_fn(api_->userdata, cell_, key, nullptr, 0);
        std::vector<double> result(size);
        if (size) api_->get_tuple_fn(api_->userdata, cell_, key, result.data(), size);
        return result;
    }
    std::vector<double> getList(const char *key) const {
        if (!queries_ || !queries_->get_list_fn || !key) return {};
        const size_t size = queries_->get_list_fn(queries_->userdata, cell_, key, nullptr, 0u);
        std::vector<double> result(size);
        if (size) queries_->get_list_fn(queries_->userdata, cell_, key, result.data(), size);
        return result;
    }

protected:
    const CC3DPythonDictionaryAPIV1 *api_;
    const CC3DPythonDictionaryQueriesAPIV1 *queries_;
    CC3DCellHandle cell_;
};

class CellAttributes : public PythonDictionary {
public:
    CellAttributes() = default;
    CellAttributes(const CC3DPythonDictionaryAPIV1 *api, CC3DCellHandle cell,
                   const CC3DPythonDictionaryQueriesAPIV1 *queries = nullptr)
        : PythonDictionary(api, cell, queries) {}
};

class SharedData : public PythonDictionary {
public:
    SharedData() = default;
    explicit SharedData(const CC3DPythonDictionaryAPIV1 *api,
                        const CC3DPythonDictionaryQueriesAPIV1 *queries = nullptr)
        : PythonDictionary(api, nullptr, queries) {}
};

class CellNeighbor {
public:
    CellNeighbor() : neighbor_(CC3DCellViewV1{}), commonSurfaceArea(0) {}

    explicit CellNeighbor(const CC3DCellNeighborViewV1 &view)
        : neighbor_(view.neighbor), commonSurfaceArea(view.commonSurfaceArea) {}

    explicit operator bool() const { return neighbor_.handle() != nullptr; }

    Cell &neighbor() { return neighbor_; }
    const Cell &neighbor() const { return neighbor_; }

    int commonSurfaceArea;

private:
    Cell neighbor_;
};

class OptionalCell {
public:
    OptionalCell() : hasValue_(false), value_(CC3DCellViewV1{}) {}

    explicit OptionalCell(const CC3DOptionalCellViewV1 &view)
        : hasValue_(view.hasCell != 0), value_(view.cell) {}

    explicit operator bool() const { return hasValue_; }

    Cell &value() { return value_; }
    const Cell &value() const { return value_; }

private:
    bool hasValue_;
    Cell value_;
};

class CellIterator {
public:
    CellIterator() : api_(nullptr), handle_(nullptr) {}

    CellIterator(const CC3DCellsAPIV1 *api, CC3DCellIteratorHandle handle)
        : api_(api), handle_(handle) {}

    CellIterator(const CellIterator &) = delete;
    CellIterator &operator=(const CellIterator &) = delete;

    CellIterator(CellIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    CellIterator &operator=(CellIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~CellIterator() { cleanup(); }

    Cell operator*() const { return Cell(api_->get_fn(api_->userdata, handle_)); }

    CellIterator &operator++() {
        api_->next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(CellSentinel) const {
        return api_ && handle_ && api_->valid_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->destroy_fn) {
            api_->destroy_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DCellsAPIV1 *api_;
    CC3DCellIteratorHandle handle_;
};

class CellTypeRange {
public:
    CellTypeRange() : api_(nullptr) {}

    CellTypeRange(const CC3DCellsAPIV1 *api, unsigned char type_id)
        : api_(api), type_ids_(1, type_id) {}

    CellTypeRange(const CC3DCellsAPIV1 *api, const std::vector<unsigned char> &type_ids)
        : api_(api), type_ids_(type_ids) {}

    CellIterator begin() const {
        if (!api_ || !api_->by_type_begin_fn) return CellIterator();
        return CellIterator(api_, api_->by_type_begin_fn(api_->userdata, type_ids_.data(), type_ids_.size()));
    }

    CellSentinel end() const { return CellSentinel(); }

private:
    const CC3DCellsAPIV1 *api_;
    std::vector<unsigned char> type_ids_;
};

class ClusterCellIterator {
public:
    ClusterCellIterator() : api_(nullptr), handle_(nullptr) {}

    ClusterCellIterator(const CC3DClusterCellsAPIV1 *api, CC3DCellIteratorHandle handle)
        : api_(api), handle_(handle) {}

    ClusterCellIterator(const ClusterCellIterator &) = delete;
    ClusterCellIterator &operator=(const ClusterCellIterator &) = delete;

    ClusterCellIterator(ClusterCellIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    ClusterCellIterator &operator=(ClusterCellIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~ClusterCellIterator() { cleanup(); }

    Cell operator*() const { return Cell(api_->cells_get_fn(api_->userdata, handle_)); }

    ClusterCellIterator &operator++() {
        api_->cells_next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(CellSentinel) const {
        return api_ && handle_ && api_->cells_valid_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->cells_destroy_fn) api_->cells_destroy_fn(api_->userdata, handle_);
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DClusterCellsAPIV1 *api_;
    CC3DCellIteratorHandle handle_;
};

class ClusterCellRange {
public:
    ClusterCellRange() : api_(nullptr), cluster_id_(0) {}

    ClusterCellRange(const CC3DClusterCellsAPIV1 *api, long cluster_id)
        : api_(api), cluster_id_(cluster_id) {}

    ClusterCellIterator begin() const {
        if (!api_ || !api_->cells_begin_fn) return ClusterCellIterator();
        return ClusterCellIterator(api_, api_->cells_begin_fn(api_->userdata, cluster_id_));
    }

    CellSentinel end() const { return CellSentinel(); }

private:
    const CC3DClusterCellsAPIV1 *api_;
    long cluster_id_;
};

class Cluster {
public:
    Cluster() : api_(nullptr), cluster_id_(0) {}

    Cluster(const CC3DClusterCellsAPIV1 *api, long cluster_id)
        : api_(api), cluster_id_(cluster_id) {}

    long id() const { return cluster_id_; }
    ClusterCellRange cells() const { return ClusterCellRange(api_, cluster_id_); }

private:
    const CC3DClusterCellsAPIV1 *api_;
    long cluster_id_;
};

class ClusterIterator {
public:
    ClusterIterator() : api_(nullptr), handle_(nullptr) {}

    ClusterIterator(const CC3DClusterCellsAPIV1 *api, CC3DClusterIteratorHandle handle)
        : api_(api), handle_(handle) {}

    ClusterIterator(const ClusterIterator &) = delete;
    ClusterIterator &operator=(const ClusterIterator &) = delete;

    ClusterIterator(ClusterIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    ClusterIterator &operator=(ClusterIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~ClusterIterator() { cleanup(); }

    Cluster operator*() const {
        return Cluster(api_, api_->clusters_get_id_fn(api_->userdata, handle_));
    }

    ClusterIterator &operator++() {
        api_->clusters_next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(CellSentinel) const {
        return api_ && handle_ && api_->clusters_valid_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->clusters_destroy_fn) api_->clusters_destroy_fn(api_->userdata, handle_);
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DClusterCellsAPIV1 *api_;
    CC3DClusterIteratorHandle handle_;
};

class ClusterRange {
public:
    explicit ClusterRange(const CC3DClusterCellsAPIV1 *api = nullptr) : api_(api) {}

    ClusterIterator begin() const {
        if (!api_ || !api_->clusters_begin_fn) return ClusterIterator();
        return ClusterIterator(api_, api_->clusters_begin_fn(api_->userdata));
    }

    CellSentinel end() const { return CellSentinel(); }

private:
    const CC3DClusterCellsAPIV1 *api_;
};

class CellNeighborIterator {
public:
    CellNeighborIterator() : api_(nullptr), handle_(nullptr) {}

    CellNeighborIterator(const CC3DCellNeighborsAPIV1 *api, CC3DCellNeighborIteratorHandle handle)
        : api_(api), handle_(handle) {}

    CellNeighborIterator(const CellNeighborIterator &) = delete;
    CellNeighborIterator &operator=(const CellNeighborIterator &) = delete;

    CellNeighborIterator(CellNeighborIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    CellNeighborIterator &operator=(CellNeighborIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~CellNeighborIterator() { cleanup(); }

    CellNeighbor operator*() const { return CellNeighbor(api_->get_fn(api_->userdata, handle_)); }

    CellNeighborIterator &operator++() {
        api_->next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(CellNeighborSentinel) const {
        return api_ && handle_ && api_->valid_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->destroy_fn) {
            api_->destroy_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DCellNeighborsAPIV1 *api_;
    CC3DCellNeighborIteratorHandle handle_;
};

class CellNeighborRange {
public:
    CellNeighborRange() : api_(nullptr), cell_(nullptr) {}

    CellNeighborRange(const CC3DCellNeighborsAPIV1 *api, CC3DCellHandle cell)
        : api_(api), cell_(cell) {}

    CellNeighborIterator begin() const {
        if (!api_ || !api_->begin_fn || !cell_) {
            return CellNeighborIterator();
        }
        return CellNeighborIterator(api_, api_->begin_fn(api_->userdata, cell_));
    }

    CellNeighborSentinel end() const { return CellNeighborSentinel(); }

private:
    const CC3DCellNeighborsAPIV1 *api_;
    CC3DCellHandle cell_;
};


class Pixel {
public:
    Pixel() : x(0), y(0), z(0) {}
    explicit Pixel(const CC3DPixelViewV1 &view) : x(view.x), y(view.y), z(view.z) {}

    int x;
    int y;
    int z;
};

class PixelIterator {
public:
    PixelIterator() : api_(nullptr), handle_(nullptr) {}

    PixelIterator(const CC3DCellPixelsAPIV1 *api, CC3DPixelIteratorHandle handle)
        : api_(api), handle_(handle) {}

    PixelIterator(const PixelIterator &) = delete;
    PixelIterator &operator=(const PixelIterator &) = delete;

    PixelIterator(PixelIterator &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    PixelIterator &operator=(PixelIterator &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~PixelIterator() { cleanup(); }

    Pixel operator*() const { return Pixel(api_->get_fn(api_->userdata, handle_)); }

    PixelIterator &operator++() {
        api_->next_fn(api_->userdata, handle_);
        return *this;
    }

    bool operator!=(PixelSentinel) const {
        return api_ && handle_ && api_->valid_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->destroy_fn) {
            api_->destroy_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DCellPixelsAPIV1 *api_;
    CC3DPixelIteratorHandle handle_;
};

class PixelRange {
public:
    PixelRange() : api_(nullptr), cell_(nullptr) {}

    PixelRange(const CC3DCellPixelsAPIV1 *api, CC3DCellHandle cell)
        : api_(api), cell_(cell) {}

    PixelIterator begin() const {
        if (!api_ || !api_->begin_fn || !cell_) {
            return PixelIterator();
        }
        return PixelIterator(api_, api_->begin_fn(api_->userdata, cell_));
    }

    PixelSentinel end() const { return PixelSentinel(); }

private:
    const CC3DCellPixelsAPIV1 *api_;
    CC3DCellHandle cell_;
};

struct EveryPixelSentinel {};

class EveryPixelIterator {
public:
    EveryPixelIterator() : dim_{0, 0, 0}, step_{1, 1, 1}, current_{0, 0, 0}, valid_(false) {}

    EveryPixelIterator(const Dim3 &dim, int step_x, int step_y, int step_z, bool valid)
        : dim_(dim), step_{step_x, step_y, step_z}, current_{0, 0, 0}, valid_(valid) {}

    Pixel operator*() const {
        Pixel result;
        result.x = current_.x;
        result.y = current_.y;
        result.z = current_.z;
        return result;
    }

    EveryPixelIterator &operator++() {
        if (!valid_) return *this;
        current_.z += step_.z;
        if (current_.z < dim_.z) return *this;
        current_.z = 0;
        current_.y += step_.y;
        if (current_.y < dim_.y) return *this;
        current_.y = 0;
        current_.x += step_.x;
        if (current_.x >= dim_.x) valid_ = false;
        return *this;
    }

    bool operator!=(EveryPixelSentinel) const { return valid_; }

private:
    Dim3 dim_;
    Dim3 step_;
    Dim3 current_;
    bool valid_;
};

class EveryPixelRange {
public:
    EveryPixelRange() : dim_{0, 0, 0}, step_{1, 1, 1}, valid_(false) {}

    EveryPixelRange(const Dim3 &dim, int step_x, int step_y, int step_z)
        : dim_(dim), step_{step_x, step_y, step_z},
          valid_(step_x > 0 && step_y > 0 && step_z > 0 && dim.x > 0 && dim.y > 0 && dim.z > 0) {}

    EveryPixelIterator begin() const {
        return EveryPixelIterator(dim_, step_.x, step_.y, step_.z, valid_);
    }

    EveryPixelSentinel end() const { return EveryPixelSentinel(); }

private:
    Dim3 dim_;
    Dim3 step_;
    bool valid_;
};

class ScalarFieldValue {
public:
    ScalarFieldValue(const CC3DScalarFieldsAPIV1 *api, CC3DFieldHandle field, int x, int y, int z)
        : api_(api), field_(field), x_(x), y_(y), z_(z) {}

    operator float() const {
        if (!api_ || !field_ || !api_->get_fn) {
            return 0.0f;
        }
        return api_->get_fn(api_->userdata, field_, x_, y_, z_);
    }

    ScalarFieldValue &operator=(float value) {
        if (api_ && field_ && api_->set_fn) {
            api_->set_fn(api_->userdata, field_, x_, y_, z_, value);
        }
        return *this;
    }

    ScalarFieldValue &operator+=(float value) {
        *this = static_cast<float>(*this) + value;
        return *this;
    }

    ScalarFieldValue &operator-=(float value) {
        *this = static_cast<float>(*this) - value;
        return *this;
    }

private:
    const CC3DScalarFieldsAPIV1 *api_;
    CC3DFieldHandle field_;
    int x_;
    int y_;
    int z_;
};

class ScalarField {
public:
    ScalarField() : api_(nullptr), field_(nullptr) {}
    ScalarField(const CC3DScalarFieldsAPIV1 *api, CC3DFieldHandle field)
        : api_(api), field_(field) {}

    explicit operator bool() const { return api_ && field_; }

    Dim3 dim() const {
        if (!api_ || !field_ || !api_->dim_fn) {
            return {0, 0, 0};
        }
        CC3DFieldDimV1 d = api_->dim_fn(api_->userdata, field_);
        return {d.x, d.y, d.z};
    }

    ScalarFieldValue operator()(int x, int y, int z) const {
        return ScalarFieldValue(api_, field_, x, y, z);
    }

private:
    const CC3DScalarFieldsAPIV1 *api_;
    CC3DFieldHandle field_;
};


class SBMLModel {
public:
    SBMLModel() : api_(nullptr), handle_(nullptr) {}

    SBMLModel(const CC3DSBMLAPIV1 *api, CC3DSBMLModelHandle handle)
        : api_(api), handle_(handle) {}

    SBMLModel(const SBMLModel &) = delete;
    SBMLModel &operator=(const SBMLModel &) = delete;

    SBMLModel(SBMLModel &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    SBMLModel &operator=(SBMLModel &&other) noexcept {
        if (this != &other) {
            cleanup();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~SBMLModel() { cleanup(); }

    explicit operator bool() const { return api_ && handle_; }

    bool has(const char *value_name) const {
        return api_ && handle_ && api_->has_value_fn && api_->has_value_fn(api_->userdata, handle_, value_name) != 0;
    }

    bool get(const char *value_name, double *value) const {
        return api_ && handle_ && value && api_->get_value_fn && api_->get_value_fn(api_->userdata, handle_, value_name, value) != 0;
    }

    double get(const char *value_name, double default_value = 0.0) const {
        double value = default_value;
        get(value_name, &value);
        return value;
    }

    bool set(const char *value_name, double value) const {
        return api_ && handle_ && api_->set_value_fn && api_->set_value_fn(api_->userdata, handle_, value_name, value) != 0;
    }

    bool timestep() const {
        return api_ && handle_ && api_->timestep_fn && api_->timestep_fn(api_->userdata, handle_) != 0;
    }

private:
    void cleanup() {
        if (api_ && handle_ && api_->destroy_model_fn) {
            api_->destroy_model_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DSBMLAPIV1 *api_;
    CC3DSBMLModelHandle handle_;
};

class AdhesionFlex {
public:
    explicit AdhesionFlex(const CC3DAdhesionFlexAPIV1 *api) : api_(api) {}

    size_t moleculeCount() const {
        return api_ && api_->molecule_count_fn ? api_->molecule_count_fn(api_->userdata) : 0u;
    }

    const char *moleculeName(size_t index) const {
        return api_ && api_->molecule_name_fn ? api_->molecule_name_fn(api_->userdata, index) : nullptr;
    }

    float get(const Cell &cell, const char *molecule_name) const {
        if (!api_ || !api_->get_density_fn || !molecule_name) {
            return 0.0f;
        }
        return api_->get_density_fn(api_->userdata, cell.handle(), molecule_name);
    }

    float get(const Cell &cell, size_t index) const {
        return api_ && api_->get_density_by_index_fn
               ? api_->get_density_by_index_fn(api_->userdata, cell.handle(), index)
               : 0.0f;
    }

    void set(const Cell &cell, const char *molecule_name, float density) const {
        if (api_ && api_->set_density_fn && molecule_name) {
            api_->set_density_fn(api_->userdata, cell.handle(), molecule_name, density);
        }
    }

    void set(const Cell &cell, size_t index, float density) const {
        if (api_ && api_->set_density_by_index_fn) {
            api_->set_density_by_index_fn(api_->userdata, cell.handle(), index, density);
        }
    }

    bool setVector(const Cell &cell, const float *densities, size_t count) const {
        return api_ && api_->set_density_vector_fn && densities
               ? api_->set_density_vector_fn(api_->userdata, cell.handle(), densities, count) != 0
               : false;
    }

    void assignVector(const Cell &cell, const float *densities, size_t count) const {
        if (api_ && api_->assign_density_vector_fn && densities) {
            api_->assign_density_vector_fn(api_->userdata, cell.handle(), densities, count);
        }
    }

    float medium(const char *molecule_name) const {
        if (!api_ || !api_->get_medium_density_fn || !molecule_name) {
            return 0.0f;
        }
        return api_->get_medium_density_fn(api_->userdata, molecule_name);
    }

    float medium(size_t index) const {
        return api_ && api_->get_medium_density_by_index_fn
               ? api_->get_medium_density_by_index_fn(api_->userdata, index)
               : 0.0f;
    }

    void setMedium(const char *molecule_name, float density) const {
        if (api_ && api_->set_medium_density_fn && molecule_name) {
            api_->set_medium_density_fn(api_->userdata, molecule_name, density);
        }
    }

    void setMedium(size_t index, float density) const {
        if (api_ && api_->set_medium_density_by_index_fn) {
            api_->set_medium_density_by_index_fn(api_->userdata, index, density);
        }
    }

    bool setMediumVector(const float *densities, size_t count) const {
        return api_ && api_->set_medium_density_vector_fn && densities
               ? api_->set_medium_density_vector_fn(api_->userdata, densities, count) != 0
               : false;
    }

    void assignMediumVector(const float *densities, size_t count) const {
        if (api_ && api_->assign_medium_density_vector_fn && densities) {
            api_->assign_medium_density_vector_fn(api_->userdata, densities, count);
        }
    }

private:
    const CC3DAdhesionFlexAPIV1 *api_;
};

class LengthConstraint {
public:
    explicit LengthConstraint(const CC3DLengthConstraintAPIV1 *api) : api_(api) {}

    void set(const Cell &cell, double lambda_length, double target_length,
             double minor_target_length = 0.0) const {
        if (api_ && api_->set_fn) {
            api_->set_fn(api_->userdata, cell.handle(), lambda_length, target_length,
                         minor_target_length);
        }
    }

    double lambda(const Cell &cell) const {
        return api_ && api_->get_lambda_fn ? api_->get_lambda_fn(api_->userdata, cell.handle()) : 0.0;
    }

    double target(const Cell &cell) const {
        return api_ && api_->get_target_fn ? api_->get_target_fn(api_->userdata, cell.handle()) : 0.0;
    }

    double minorTarget(const Cell &cell) const {
        return api_ && api_->get_minor_target_fn ? api_->get_minor_target_fn(api_->userdata, cell.handle()) : 0.0;
    }

private:
    const CC3DLengthConstraintAPIV1 *api_;
};

class FocalPointPlasticity {
public:
    explicit FocalPointPlasticity(
        const CC3DFocalPointPlasticityAPIV1 *api,
        const CC3DFocalPointPlasticityQueriesAPIV1 *queries = nullptr
    ) : api_(api), queries_(queries) {}

    void create(const Cell &cell1, const Cell &cell2, double lambda,
                double target_distance = 0.0, double max_distance = 0.0) const {
        if (api_ && api_->create_link_fn) api_->create_link_fn(api_->userdata, cell1.handle(), cell2.handle(), lambda, target_distance, max_distance);
    }
    void createInternal(const Cell &cell1, const Cell &cell2, double lambda,
                        double target_distance = 0.0, double max_distance = 0.0) const {
        if (api_ && api_->create_internal_link_fn) api_->create_internal_link_fn(api_->userdata, cell1.handle(), cell2.handle(), lambda, target_distance, max_distance);
    }
    void erase(const Cell &cell1, const Cell &cell2) const {
        if (api_ && api_->delete_link_fn) api_->delete_link_fn(api_->userdata, cell1.handle(), cell2.handle());
    }
    void eraseInternal(const Cell &cell1, const Cell &cell2) const {
        if (api_ && api_->delete_internal_link_fn) api_->delete_internal_link_fn(api_->userdata, cell1.handle(), cell2.handle());
    }
    void set(const Cell &cell1, const Cell &cell2, double lambda,
             double target_distance = 0.0, double max_distance = 0.0) const {
        if (api_ && api_->set_link_fn) api_->set_link_fn(api_->userdata, cell1.handle(), cell2.handle(), lambda, target_distance, max_distance);
    }
    double lambda(const Cell &cell1, const Cell &cell2) const {
        return api_ && api_->get_lambda_fn ? api_->get_lambda_fn(api_->userdata, cell1.handle(), cell2.handle()) : 0.0;
    }
    double target(const Cell &cell1, const Cell &cell2) const {
        return api_ && api_->get_target_fn ? api_->get_target_fn(api_->userdata, cell1.handle(), cell2.handle()) : 0.0;
    }
    int createAnchor(const Cell &cell, double lambda, double target_distance = 0.0,
                     double max_distance = 100000.0, float x = 0, float y = 0, float z = 0) const {
        return api_ && api_->create_anchor_fn ? api_->create_anchor_fn(api_->userdata, cell.handle(), lambda, target_distance, max_distance, x, y, z) : -1;
    }
    void eraseAnchor(const Cell &cell, int anchor_id) const {
        if (api_ && api_->delete_anchor_fn) api_->delete_anchor_fn(api_->userdata, cell.handle(), anchor_id);
    }
    void setAnchor(const Cell &cell, int anchor_id, double lambda, double target_distance = 0.0,
                   double max_distance = 100000.0, float x = -1, float y = -1, float z = -1) const {
        if (api_ && api_->set_anchor_fn) api_->set_anchor_fn(api_->userdata, cell.handle(), anchor_id, lambda, target_distance, max_distance, x, y, z);
    }

    class Link {
    public:
        Link() : view_{} {}
        explicit Link(const CC3DFPPLinkViewV1 &view) : view_(view) {}
        explicit operator bool() const { return view_.valid != 0; }
        Cell initiator() const { return Cell(view_.initiator); }
        Cell initiated() const { return Cell(view_.initiated); }
        Cell otherCell(const Cell &cell) const {
            if (cell.handle() == view_.initiator.handle) return initiated();
            if (cell.handle() == view_.initiated.handle) return initiator();
            return Cell(CC3DCellViewV1{});
        }
        int anchorId() const { return view_.anchorId; }
        CC3DFPPLinkHandle handle() const { return view_.handle; }
    private:
        CC3DFPPLinkViewV1 view_;
    };

    class LinkIterator {
    public:
        LinkIterator() : queries_(nullptr), handle_(nullptr), kind_(0) {}
        LinkIterator(const CC3DFocalPointPlasticityQueriesAPIV1 *api, CC3DFPPLinkIteratorHandle handle, int kind)
            : queries_(api), handle_(handle), kind_(kind) {}
        LinkIterator(const LinkIterator &) = delete;
        LinkIterator &operator=(const LinkIterator &) = delete;
        LinkIterator(LinkIterator &&other) noexcept
            : queries_(other.queries_), handle_(other.handle_), kind_(other.kind_) {
            other.queries_ = nullptr; other.handle_ = nullptr;
        }
        LinkIterator &operator=(LinkIterator &&other) noexcept {
            if (this != &other) { cleanup(); queries_ = other.queries_; handle_ = other.handle_; kind_ = other.kind_; other.queries_ = nullptr; other.handle_ = nullptr; }
            return *this;
        }
        ~LinkIterator() { cleanup(); }
        Link operator*() const { return Link(get()); }
        LinkIterator &operator++() { if (valid()) next(); return *this; }
        bool operator!=(CellSentinel) const { return valid(); }
    private:
        CC3DFPPLinkViewV1 get() const {
            if (!queries_ || !handle_) return CC3DFPPLinkViewV1{};
            switch (kind_) {
                case 1: return queries_->internal_links_get_fn(queries_->userdata, handle_);
                case 2: return queries_->anchors_get_fn(queries_->userdata, handle_);
                default: return queries_->links_get_fn(queries_->userdata, handle_);
            }
        }
        bool valid() const {
            if (!queries_ || !handle_) return false;
            switch (kind_) {
                case 1: return queries_->internal_links_valid_fn(queries_->userdata, handle_) != 0;
                case 2: return queries_->anchors_valid_fn(queries_->userdata, handle_) != 0;
                default: return queries_->links_valid_fn(queries_->userdata, handle_) != 0;
            }
        }
        void next() {
            switch (kind_) {
                case 1: queries_->internal_links_next_fn(queries_->userdata, handle_); break;
                case 2: queries_->anchors_next_fn(queries_->userdata, handle_); break;
                default: queries_->links_next_fn(queries_->userdata, handle_); break;
            }
        }
        void cleanup() {
            if (queries_ && handle_) {
                switch (kind_) {
                    case 1: queries_->internal_links_destroy_fn(queries_->userdata, handle_); break;
                    case 2: queries_->anchors_destroy_fn(queries_->userdata, handle_); break;
                    default: queries_->links_destroy_fn(queries_->userdata, handle_); break;
                }
            }
            queries_ = nullptr; handle_ = nullptr;
        }
        const CC3DFocalPointPlasticityQueriesAPIV1 *queries_;
        CC3DFPPLinkIteratorHandle handle_;
        int kind_;
    };

    class LinkRange {
    public:
        LinkRange(const CC3DFocalPointPlasticityQueriesAPIV1 *api, CC3DCellHandle cell, int kind)
            : queries_(api), cell_(cell), kind_(kind) {}
        LinkIterator begin() const {
            if (!queries_) return LinkIterator();
            CC3DFPPLinkIteratorHandle handle = nullptr;
            if (kind_ == 1) handle = queries_->internal_links_begin_fn(queries_->userdata, cell_);
            else if (kind_ == 2) handle = queries_->anchors_begin_fn(queries_->userdata, cell_);
            else handle = queries_->links_begin_fn(queries_->userdata, cell_);
            return LinkIterator(queries_, handle, kind_);
        }
        CellSentinel end() const { return CellSentinel(); }
    private:
        const CC3DFocalPointPlasticityQueriesAPIV1 *queries_;
        CC3DCellHandle cell_;
        int kind_;
    };

    class CellIterator {
    public:
        CellIterator() : queries_(nullptr), handle_(nullptr), internal_(false) {}
        CellIterator(const CC3DFocalPointPlasticityQueriesAPIV1 *api, CC3DFPPCellIteratorHandle handle, bool internal)
            : queries_(api), handle_(handle), internal_(internal) {}
        CellIterator(const CellIterator &) = delete;
        CellIterator &operator=(const CellIterator &) = delete;
        CellIterator(CellIterator &&other) noexcept : queries_(other.queries_), handle_(other.handle_), internal_(other.internal_) { other.queries_ = nullptr; other.handle_ = nullptr; }
        CellIterator &operator=(CellIterator &&other) noexcept { if (this != &other) { cleanup(); queries_ = other.queries_; handle_ = other.handle_; internal_ = other.internal_; other.queries_ = nullptr; other.handle_ = nullptr; } return *this; }
        ~CellIterator() { cleanup(); }
        Cell operator*() const { return Cell(get()); }
        CellIterator &operator++() { if (valid()) next(); return *this; }
        bool operator!=(CellSentinel) const { return valid(); }
    private:
        CC3DCellViewV1 get() const { return internal_ ? queries_->internal_linked_cells_get_fn(queries_->userdata, handle_) : queries_->linked_cells_get_fn(queries_->userdata, handle_); }
        bool valid() const { return queries_ && handle_ && (internal_ ? queries_->internal_linked_cells_valid_fn(queries_->userdata, handle_) : queries_->linked_cells_valid_fn(queries_->userdata, handle_)) != 0; }
        void next() { if (internal_) queries_->internal_linked_cells_next_fn(queries_->userdata, handle_); else queries_->linked_cells_next_fn(queries_->userdata, handle_); }
        void cleanup() { if (queries_ && handle_) { if (internal_) queries_->internal_linked_cells_destroy_fn(queries_->userdata, handle_); else queries_->linked_cells_destroy_fn(queries_->userdata, handle_); } queries_ = nullptr; handle_ = nullptr; }
        const CC3DFocalPointPlasticityQueriesAPIV1 *queries_;
        CC3DFPPCellIteratorHandle handle_;
        bool internal_;
    };

    class CellRange {
    public:
        CellRange(const CC3DFocalPointPlasticityQueriesAPIV1 *api, CC3DCellHandle cell, bool internal)
            : queries_(api), cell_(cell), internal_(internal) {}
        CellIterator begin() const {
            if (!queries_) return CellIterator();
            auto handle = internal_ ? queries_->internal_linked_cells_begin_fn(queries_->userdata, cell_) : queries_->linked_cells_begin_fn(queries_->userdata, cell_);
            return CellIterator(queries_, handle, internal_);
        }
        CellSentinel end() const { return CellSentinel(); }
    private:
        const CC3DFocalPointPlasticityQueriesAPIV1 *queries_;
        CC3DCellHandle cell_;
        bool internal_;
    };

    LinkRange links() const { return LinkRange(queries_, nullptr, 0); }
    LinkRange links(const Cell &cell) const { return LinkRange(queries_, cell.handle(), 0); }
    LinkRange internalLinks() const { return LinkRange(queries_, nullptr, 1); }
    LinkRange internalLinks(const Cell &cell) const { return LinkRange(queries_, cell.handle(), 1); }
    LinkRange anchors() const { return LinkRange(queries_, nullptr, 2); }
    LinkRange anchors(const Cell &cell) const { return LinkRange(queries_, cell.handle(), 2); }
    Link findLink(const Cell &cell1, const Cell &cell2) const { return queries_ && queries_->find_link_fn ? Link(queries_->find_link_fn(queries_->userdata, cell1.handle(), cell2.handle())) : Link(); }
    Link findInternalLink(const Cell &cell1, const Cell &cell2) const { return queries_ && queries_->find_internal_link_fn ? Link(queries_->find_internal_link_fn(queries_->userdata, cell1.handle(), cell2.handle())) : Link(); }
    size_t linkCount() const { return queries_ && queries_->link_count_fn ? queries_->link_count_fn(queries_->userdata, nullptr) : 0u; }
    size_t linkCount(const Cell &cell) const { return queries_ && queries_->link_count_fn ? queries_->link_count_fn(queries_->userdata, cell.handle()) : 0u; }
    size_t internalLinkCount() const { return queries_ && queries_->internal_link_count_fn ? queries_->internal_link_count_fn(queries_->userdata, nullptr) : 0u; }
    size_t internalLinkCount(const Cell &cell) const { return queries_ && queries_->internal_link_count_fn ? queries_->internal_link_count_fn(queries_->userdata, cell.handle()) : 0u; }
    size_t anchorCount() const { return queries_ && queries_->anchor_count_fn ? queries_->anchor_count_fn(queries_->userdata, nullptr) : 0u; }
    size_t anchorCount(const Cell &cell) const { return queries_ && queries_->anchor_count_fn ? queries_->anchor_count_fn(queries_->userdata, cell.handle()) : 0u; }
    CellRange linkedCells(const Cell &cell) const { return CellRange(queries_, cell.handle(), false); }
    CellRange internalLinkedCells(const Cell &cell) const { return CellRange(queries_, cell.handle(), true); }
private:
    const CC3DFocalPointPlasticityAPIV1 *api_;
    const CC3DFocalPointPlasticityQueriesAPIV1 *queries_;
};

class SharedScalarFieldValue {
public:
    SharedScalarFieldValue(const CC3DSharedScalarFieldsAPIV1 *api, CC3DSharedFieldHandle field, int x, int y, int z, bool raw)
        : api_(api), field_(field), x_(x), y_(y), z_(z), raw_(raw) {}

    operator double() const {
        return api_ && field_ && api_->get_fn ? api_->get_fn(api_->userdata, field_, x_, y_, z_, raw_ ? 1u : 0u) : 0.0;
    }

    SharedScalarFieldValue &operator=(double value) {
        if (api_ && field_ && api_->set_fn) api_->set_fn(api_->userdata, field_, x_, y_, z_, value, raw_ ? 1u : 0u);
        return *this;
    }

private:
    const CC3DSharedScalarFieldsAPIV1 *api_;
    CC3DSharedFieldHandle field_;
    int x_, y_, z_;
    bool raw_;
};

class SharedScalarField {
public:
    SharedScalarField() : api_(nullptr), field_(nullptr), raw_(false) {}
    SharedScalarField(const CC3DSharedScalarFieldsAPIV1 *api, CC3DSharedFieldHandle field, bool raw = false)
        : api_(api), field_(field), raw_(raw) {}

    explicit operator bool() const { return api_ && field_; }

    SharedScalarField raw() const { return SharedScalarField(api_, field_, true); }

    Dim3 dim() const {
        if (!api_ || !field_ || !api_->dim_fn) return {0, 0, 0};
        CC3DFieldDimV1 d = api_->dim_fn(api_->userdata, field_, raw_ ? 1u : 0u);
        return {d.x, d.y, d.z};
    }

    double get(int x, int y, int z) const {
        return api_ && field_ && api_->get_fn ? api_->get_fn(api_->userdata, field_, x, y, z, raw_ ? 1u : 0u) : 0.0;
    }

    bool set(int x, int y, int z, double value) const {
        return api_ && field_ && api_->set_fn && api_->set_fn(api_->userdata, field_, x, y, z, value, raw_ ? 1u : 0u);
    }

    SharedScalarFieldValue at(int x, int y, int z) const {
        return SharedScalarFieldValue(api_, field_, x, y, z, raw_);
    }

private:
    const CC3DSharedScalarFieldsAPIV1 *api_;
    CC3DSharedFieldHandle field_;
    bool raw_;
};

class SharedVectorField {
public:
    SharedVectorField() : api_(nullptr), field_(nullptr), raw_(false) {}
    SharedVectorField(const CC3DSharedVectorFieldsAPIV1 *api, CC3DSharedFieldHandle field, bool raw = false)
        : api_(api), field_(field), raw_(raw) {}

    explicit operator bool() const { return api_ && field_; }
    SharedVectorField raw() const { return SharedVectorField(api_, field_, true); }

    Dim3 dim() const {
        if (!api_ || !field_ || !api_->dim_fn) return {0, 0, 0};
        CC3DFieldDimV1 d = api_->dim_fn(api_->userdata, field_, raw_ ? 1u : 0u);
        return {d.x, d.y, d.z};
    }

    bool get(int x, int y, int z, double &vx, double &vy, double &vz) const {
        return api_ && field_ && api_->get_fn && api_->get_fn(api_->userdata, field_, x, y, z, &vx, &vy, &vz, raw_ ? 1u : 0u);
    }

    bool set(int x, int y, int z, double vx, double vy, double vz) const {
        return api_ && field_ && api_->set_fn && api_->set_fn(api_->userdata, field_, x, y, z, vx, vy, vz, raw_ ? 1u : 0u);
    }

private:
    const CC3DSharedVectorFieldsAPIV1 *api_;
    CC3DSharedFieldHandle field_;
    bool raw_;
};

class ClusterSurface {
public:
    explicit ClusterSurface(const CC3DClusterSurfaceAPIV1 *api) : api_(api) {}

    void setTargetAndLambdaClusterSurface(const Cell &cell, float target_cluster_surface,
                                          float lambda_cluster_surface) const {
        if (api_ && api_->set_fn) {
            api_->set_fn(api_->userdata, cell.handle(), target_cluster_surface, lambda_cluster_surface);
        }
    }

    std::pair<float, float> getTargetAndLambdaClusterSurface(const Cell &cell) const {
        std::pair<float, float> values{0.0f, 0.0f};
        if (api_ && api_->get_fn) {
            api_->get_fn(api_->userdata, cell.handle(), &values.first, &values.second);
        }
        return values;
    }

private:
    const CC3DClusterSurfaceAPIV1 *api_;
};

class ClusterSurfaceTracker {
public:
    explicit ClusterSurfaceTracker(const CC3DClusterSurfaceTrackerAPIV1 *api) : api_(api) {}

    void updateClusterSurface(long cluster_id) const {
        if (api_ && api_->update_cluster_surface_fn) {
            api_->update_cluster_surface_fn(api_->userdata, cluster_id);
        }
    }

    bool reassignClusterId(const Cell &cell, long cluster_id) const {
        return api_ && api_->reassign_cluster_id_fn
            ? api_->reassign_cluster_id_fn(api_->userdata, cell.handle(), cluster_id) != 0
            : false;
    }

private:
    const CC3DClusterSurfaceTrackerAPIV1 *api_;
};

class Mitosis {
public:
    explicit Mitosis(const CC3DMitosisAPIV1 *api) : api_(api) {}

    void setParentChildPositionFlag(int flag) const {
        if (api_ && api_->set_parent_child_position_fn) api_->set_parent_child_position_fn(api_->userdata, flag);
    }
    int parentChildPositionFlag() const {
        return api_ && api_->get_parent_child_position_fn ? api_->get_parent_child_position_fn(api_->userdata) : 0;
    }
    bool divideRandom(const Cell &cell) const { return divide(api_ ? api_->divide_random_fn : nullptr, cell); }
    bool divideOrientation(const Cell &cell, double nx, double ny, double nz) const {
        return api_ && api_->divide_orientation_fn ? api_->divide_orientation_fn(api_->userdata, cell.handle(), nx, ny, nz) != 0 : false;
    }
    bool divideAlongMajorAxis(const Cell &cell) const { return divide(api_ ? api_->divide_major_axis_fn : nullptr, cell); }
    bool divideAlongMinorAxis(const Cell &cell) const { return divide(api_ ? api_->divide_minor_axis_fn : nullptr, cell); }
    bool divideClusterRandom(long cluster_id) const { return api_ && api_->divide_cluster_random_fn ? api_->divide_cluster_random_fn(api_->userdata, cluster_id) != 0 : false; }
    bool divideClusterOrientation(long cluster_id, double nx, double ny, double nz) const {
        return api_ && api_->divide_cluster_orientation_fn ? api_->divide_cluster_orientation_fn(api_->userdata, cluster_id, nx, ny, nz) != 0 : false;
    }
    bool divideClusterAlongMajorAxis(long cluster_id) const { return api_ && api_->divide_cluster_major_axis_fn ? api_->divide_cluster_major_axis_fn(api_->userdata, cluster_id) != 0 : false; }
    bool divideClusterAlongMinorAxis(long cluster_id) const { return api_ && api_->divide_cluster_minor_axis_fn ? api_->divide_cluster_minor_axis_fn(api_->userdata, cluster_id) != 0 : false; }
    Cell parentCell() const { return api_ && api_->parent_cell_fn ? Cell(api_->parent_cell_fn(api_->userdata)) : Cell(CC3DCellViewV1{}); }
    Cell childCell() const { return api_ && api_->child_cell_fn ? Cell(api_->child_cell_fn(api_->userdata)) : Cell(CC3DCellViewV1{}); }
    void cloneAttributes(const Cell &source, const Cell &target) const {
        if (api_ && api_->clone_attributes_fn) api_->clone_attributes_fn(api_->userdata, source.handle(), target.handle());
    }

private:
    typedef uint8_t (*DivideFn)(void *, CC3DCellHandle);
    bool divide(DivideFn fn, const Cell &cell) const { return api_ && fn ? fn(api_->userdata, cell.handle()) != 0 : false; }
    const CC3DMitosisAPIV1 *api_;
};

class OrientedGrowth {
public:
    explicit OrientedGrowth(const CC3DOrientedGrowthAPIV1 *api) : api_(api) {}

    void setConstraintWidth(const Cell &cell, float constraint) const {
        if (api_ && api_->set_constraint_width_fn) api_->set_constraint_width_fn(api_->userdata, cell.handle(), constraint);
    }
    void setElongationAxis(const Cell &cell, float x, float y) const {
        if (api_ && api_->set_elongation_axis_fn) api_->set_elongation_axis_fn(api_->userdata, cell.handle(), x, y);
    }
    void setElongationEnabled(const Cell &cell, bool enabled) const {
        if (api_ && api_->set_elongation_enabled_fn) api_->set_elongation_enabled_fn(api_->userdata, cell.handle(), enabled ? 1u : 0u);
    }
    float getElongationAxis_X(const Cell &cell) const {
        return api_ && api_->get_elongation_axis_x_fn ? api_->get_elongation_axis_x_fn(api_->userdata, cell.handle()) : 0.0f;
    }
    float getElongationAxis_Y(const Cell &cell) const {
        return api_ && api_->get_elongation_axis_y_fn ? api_->get_elongation_axis_y_fn(api_->userdata, cell.handle()) : 0.0f;
    }
    bool getElongationEnabled(const Cell &cell) const {
        return api_ && api_->get_elongation_enabled_fn && api_->get_elongation_enabled_fn(api_->userdata, cell.handle()) != 0;
    }

    float elongationAxisX(const Cell &cell) const { return getElongationAxis_X(cell); }
    float elongationAxisY(const Cell &cell) const { return getElongationAxis_Y(cell); }
    bool elongationEnabled(const Cell &cell) const { return getElongationEnabled(cell); }

private:
    const CC3DOrientedGrowthAPIV1 *api_;
};

class MomentOfInertia {
public:
    explicit MomentOfInertia(const CC3DMomentOfInertiaAPIV1 *api) : api_(api) {}

    std::vector<double> getSemiaxes(const Cell &cell) const { return get(cell, api_ ? api_->get_semiaxes_fn : nullptr); }
    std::vector<double> getSemiaxesXY(const Cell &cell) const { return get(cell, api_ ? api_->get_semiaxes_xy_fn : nullptr); }
    std::vector<double> getSemiaxesXZ(const Cell &cell) const { return get(cell, api_ ? api_->get_semiaxes_xz_fn : nullptr); }
    std::vector<double> getSemiaxesYZ(const Cell &cell) const { return get(cell, api_ ? api_->get_semiaxes_yz_fn : nullptr); }
    std::vector<double> getSemiaxes3D(const Cell &cell) const { return get(cell, api_ ? api_->get_semiaxes_3d_fn : nullptr); }

private:
    using Getter = size_t (*)(void *, CC3DCellHandle, double *, size_t);

    std::vector<double> get(const Cell &cell, Getter getter) const {
        if (!api_ || !getter) return {};
        std::vector<double> values(3, 0.0);
        const size_t count = getter(api_->userdata, cell.handle(), values.data(), values.size());
        values.resize(count > values.size() ? values.size() : count);
        return values;
    }

    const CC3DMomentOfInertiaAPIV1 *api_;
};

class PolarizationVector {
public:
    explicit PolarizationVector(const CC3DPolarizationVectorAPIV1 *api) : api_(api) {}
    void set(const Cell &cell, float x, float y, float z) const {
        if (api_ && api_->set_fn) api_->set_fn(api_->userdata, cell.handle(), x, y, z);
    }
    void setPolarizationVector(const Cell &cell, float x, float y, float z) const { set(cell, x, y, z); }
    bool get(const Cell &cell, float *x, float *y, float *z) const {
        return api_ && api_->get_fn && x && y && z
               ? api_->get_fn(api_->userdata, cell.handle(), x, y, z) != 0 : false;
    }
    std::vector<float> get(const Cell &cell) const {
        float x = 0, y = 0, z = 0;
        get(cell, &x, &y, &z);
        return {x, y, z};
    }
    std::vector<float> getPolarizationVector(const Cell &cell) const { return get(cell); }
private:
    const CC3DPolarizationVectorAPIV1 *api_;
};

class CellOrientation {
public:
    explicit CellOrientation(const CC3DCellOrientationAPIV1 *api) : api_(api) {}
    void setLambda(const Cell &cell, double value) const {
        if (api_ && api_->set_lambda_fn) api_->set_lambda_fn(api_->userdata, cell.handle(), value);
    }
    double lambda(const Cell &cell) const {
        return api_ && api_->get_lambda_fn ? api_->get_lambda_fn(api_->userdata, cell.handle()) : 0.0;
    }
    double getLambda(const Cell &cell) const { return lambda(cell); }
private:
    const CC3DCellOrientationAPIV1 *api_;
};

class ChemotaxisData {
public:
    ChemotaxisData() : api_{} {}
    explicit ChemotaxisData(const CC3DChemotaxisDataAPIV1 &api) : api_(api) {}
    explicit operator bool() const { return api_.userdata != nullptr; }
    float lambda() const { return api_.userdata && api_.get_lambda_fn ? api_.get_lambda_fn(api_.userdata) : 0.0f; }
    void setLambda(float value) const { if (api_.userdata && api_.set_lambda_fn) api_.set_lambda_fn(api_.userdata, value); }
    float saturationCoef() const { return api_.userdata && api_.get_saturation_fn ? api_.get_saturation_fn(api_.userdata) : 0.0f; }
    void setSaturationCoef(float value) const { if (api_.userdata && api_.set_saturation_fn) api_.set_saturation_fn(api_.userdata, value); }
    const char *formulaName() const { return api_.userdata && api_.get_formula_name_fn ? api_.get_formula_name_fn(api_.userdata) : nullptr; }
    bool setFormulaName(const char *name) const { return api_.userdata && api_.set_formula_name_fn && name ? api_.set_formula_name_fn(api_.userdata, name) != 0 : false; }
    std::vector<int> chemotactTowardsTypes() const {
        std::vector<int> result;
        if (!api_.userdata || !api_.get_toward_count_fn || !api_.get_toward_type_fn) return result;
        for (size_t i = 0; i < api_.get_toward_count_fn(api_.userdata); ++i) result.push_back(api_.get_toward_type_fn(api_.userdata, i));
        return result;
    }
    void assignChemotactTowardsTypes(const std::vector<int> &types) const {
        if (api_.userdata && api_.assign_toward_types_fn) api_.assign_toward_types_fn(api_.userdata, types.data(), types.size());
    }
    void assignChemotactTowardsVectorTypes(const std::vector<int> &types) const { assignChemotactTowardsTypes(types); }
    std::vector<int> getChemotactTowardsVectorTypes() const { return chemotactTowardsTypes(); }
    void setChemotaxisFormulaByName(const char *name) const { setFormulaName(name); }
private:
    CC3DChemotaxisDataAPIV1 api_;
};

class Chemotaxis {
public:
    explicit Chemotaxis(const CC3DChemotaxisAPIV1 *api) : api_(api) {}
    std::vector<std::string> fieldNames(const Cell &cell) const {
        std::vector<std::string> result;
        if (!api_ || !api_->field_count_fn || !api_->field_name_fn) return result;
        for (size_t i = 0; i < api_->field_count_fn(api_->userdata, cell.handle()); ++i) {
            const char *name = api_->field_name_fn(api_->userdata, cell.handle(), i);
            if (name) result.emplace_back(name);
        }
        return result;
    }
    std::vector<std::string> getFieldNamesWithChemotaxisData(const Cell &cell) const { return fieldNames(cell); }
    ChemotaxisData add(const Cell &cell, const char *fieldName) const {
        return api_ && api_->add_data_fn && fieldName ? ChemotaxisData(api_->add_data_fn(api_->userdata, cell.handle(), fieldName)) : ChemotaxisData(CC3DChemotaxisDataAPIV1{});
    }
    ChemotaxisData addChemotaxisData(const Cell &cell, const char *fieldName) const { return add(cell, fieldName); }
    ChemotaxisData get(const Cell &cell, const char *fieldName) const {
        return api_ && api_->get_data_fn && fieldName ? ChemotaxisData(api_->get_data_fn(api_->userdata, cell.handle(), fieldName)) : ChemotaxisData(CC3DChemotaxisDataAPIV1{});
    }
    ChemotaxisData getChemotaxisData(const Cell &cell, const char *fieldName) const { return get(cell, fieldName); }
private:
    const CC3DChemotaxisAPIV1 *api_;
};

struct FieldSecretorResult {
    bool success;
    float totalAmount;
};

class FieldSecretor {
public:
    explicit FieldSecretor(const CC3DSecretionAPIV1 *api, const char *field = nullptr) : api_(api), field_(field ? field : "") {}
    bool secreteInsideCell(const Cell &cell, const char *field, float amount) const { return call(api_ ? api_->secrete_inside_fn : nullptr, cell, field, amount); }
    FieldSecretorResult secreteInsideCellTotalCount(const Cell &cell, const char *field, float amount) const { return total(api_ ? api_->secrete_inside_total_fn : nullptr, cell, field, amount); }
    bool secreteInsideCellConstantConcentration(const Cell &cell, const char *field, float amount) const { return call(api_ ? api_->secrete_inside_constant_fn : nullptr, cell, field, amount); }
    bool secreteInsideCellAtBoundary(const Cell &cell, const char *field, float amount) const { return call(api_ ? api_->secrete_inside_boundary_fn : nullptr, cell, field, amount); }
    FieldSecretorResult secreteInsideCellAtBoundaryTotalCount(const Cell &cell, const char *field, float amount) const { return total(api_ ? api_->secrete_inside_boundary_total_fn : nullptr, cell, field, amount); }
    bool secreteInsideCellAtBoundaryOnContactWith(const Cell &cell, const char *field, float amount, const std::vector<unsigned char> &types) const { return api_ && api_->secrete_inside_boundary_contact_fn ? api_->secrete_inside_boundary_contact_fn(api_->userdata, cell.handle(), field, amount, types.data(), types.size()) != 0 : false; }
    FieldSecretorResult secreteInsideCellAtBoundaryOnContactWithTotalCount(const Cell &cell, const char *field, float amount, const std::vector<unsigned char> &types) const { return api_ && api_->secrete_inside_boundary_contact_total_fn ? result(api_->secrete_inside_boundary_contact_total_fn(api_->userdata, cell.handle(), field, amount, types.data(), types.size())) : FieldSecretorResult{false, 0}; }
    bool secreteOutsideCellAtBoundary(const Cell &cell, const char *field, float amount) const { return call(api_ ? api_->secrete_outside_boundary_fn : nullptr, cell, field, amount); }
    FieldSecretorResult secreteOutsideCellAtBoundaryTotalCount(const Cell &cell, const char *field, float amount) const { return total(api_ ? api_->secrete_outside_boundary_total_fn : nullptr, cell, field, amount); }
    bool secreteOutsideCellAtBoundaryOnContactWith(const Cell &cell, const char *field, float amount, const std::vector<unsigned char> &types) const { return api_ && api_->secrete_outside_boundary_contact_fn ? api_->secrete_outside_boundary_contact_fn(api_->userdata, cell.handle(), field, amount, types.data(), types.size()) != 0 : false; }
    FieldSecretorResult secreteOutsideCellAtBoundaryOnContactWithTotalCount(const Cell &cell, const char *field, float amount, const std::vector<unsigned char> &types) const { return api_ && api_->secrete_outside_boundary_contact_total_fn ? result(api_->secrete_outside_boundary_contact_total_fn(api_->userdata, cell.handle(), field, amount, types.data(), types.size())) : FieldSecretorResult{false, 0}; }
    bool secreteInsideCellAtCOM(const Cell &cell, const char *field, float amount) const { return call(api_ ? api_->secrete_com_fn : nullptr, cell, field, amount); }
    FieldSecretorResult secreteInsideCellAtCOMTotalCount(const Cell &cell, const char *field, float amount) const { return total(api_ ? api_->secrete_com_total_fn : nullptr, cell, field, amount); }
    bool uptakeInsideCell(const Cell &cell, const char *field, float maxUptake, float relativeUptake) const { return api_ && api_->uptake_inside_fn ? api_->uptake_inside_fn(api_->userdata, cell.handle(), field, maxUptake, relativeUptake) != 0 : false; }
    FieldSecretorResult uptakeInsideCellTotalCount(const Cell &cell, const char *field, float maxUptake, float relativeUptake) const { return api_ && api_->uptake_inside_total_fn ? result(api_->uptake_inside_total_fn(api_->userdata, cell.handle(), field, maxUptake, relativeUptake)) : FieldSecretorResult{false, 0}; }
    bool uptakeInsideCellAtBoundary(const Cell &cell, const char *field, float maxUptake, float relativeUptake) const { return api_ && api_->uptake_boundary_fn ? api_->uptake_boundary_fn(api_->userdata, cell.handle(), field, maxUptake, relativeUptake) != 0 : false; }
    FieldSecretorResult uptakeInsideCellAtBoundaryTotalCount(const Cell &cell, const char *field, float maxUptake, float relativeUptake) const { return api_ && api_->uptake_boundary_total_fn ? result(api_->uptake_boundary_total_fn(api_->userdata, cell.handle(), field, maxUptake, relativeUptake)) : FieldSecretorResult{false, 0}; }
    bool uptakeOutsideCellAtBoundary(const Cell &cell, const char *field, float maxUptake, float relativeUptake) const { return api_ && api_->uptake_outside_boundary_fn ? api_->uptake_outside_boundary_fn(api_->userdata, cell.handle(), field, maxUptake, relativeUptake) != 0 : false; }
    FieldSecretorResult uptakeOutsideCellAtBoundaryTotalCount(const Cell &cell, const char *field, float maxUptake, float relativeUptake) const { return api_ && api_->uptake_outside_boundary_total_fn ? result(api_->uptake_outside_boundary_total_fn(api_->userdata, cell.handle(), field, maxUptake, relativeUptake)) : FieldSecretorResult{false, 0}; }
    float amountSeenByCell(const Cell &cell, const char *field) const { return api_ && api_->amount_seen_fn ? api_->amount_seen_fn(api_->userdata, cell.handle(), field) : 0.0f; }
    float totalFieldIntegral(const char *field) const { return api_ && api_->field_integral_fn ? api_->field_integral_fn(api_->userdata, field) : 0.0f; }
    // Field-bound overloads mirror Python's get_field_secretor(field_name). The
    // overloads taking an explicit field remain available for source compatibility.
    bool secreteInsideCell(const Cell &cell, float amount) const { return secreteInsideCell(cell, field_.c_str(), amount); }
    FieldSecretorResult secreteInsideCellTotalCount(const Cell &cell, float amount) const { return secreteInsideCellTotalCount(cell, field_.c_str(), amount); }
    bool secreteInsideCellConstantConcentration(const Cell &cell, float amount) const { return secreteInsideCellConstantConcentration(cell, field_.c_str(), amount); }
    FieldSecretorResult secreteInsideCellConstantConcentrationTotalCount(const Cell &cell, float amount) const { return total(api_ ? api_->secrete_inside_constant_total_fn : nullptr, cell, field_.c_str(), amount); }
    bool secreteInsideCellAtBoundary(const Cell &cell, float amount) const { return secreteInsideCellAtBoundary(cell, field_.c_str(), amount); }
    FieldSecretorResult secreteInsideCellAtBoundaryTotalCount(const Cell &cell, float amount) const { return secreteInsideCellAtBoundaryTotalCount(cell, field_.c_str(), amount); }
    bool secreteInsideCellAtBoundaryOnContactWith(const Cell &cell, float amount, const std::vector<unsigned char> &types) const { return secreteInsideCellAtBoundaryOnContactWith(cell, field_.c_str(), amount, types); }
    FieldSecretorResult secreteInsideCellAtBoundaryOnContactWithTotalCount(const Cell &cell, float amount, const std::vector<unsigned char> &types) const { return secreteInsideCellAtBoundaryOnContactWithTotalCount(cell, field_.c_str(), amount, types); }
    bool secreteOutsideCellAtBoundary(const Cell &cell, float amount) const { return secreteOutsideCellAtBoundary(cell, field_.c_str(), amount); }
    FieldSecretorResult secreteOutsideCellAtBoundaryTotalCount(const Cell &cell, float amount) const { return secreteOutsideCellAtBoundaryTotalCount(cell, field_.c_str(), amount); }
    bool secreteOutsideCellAtBoundaryOnContactWith(const Cell &cell, float amount, const std::vector<unsigned char> &types) const { return secreteOutsideCellAtBoundaryOnContactWith(cell, field_.c_str(), amount, types); }
    FieldSecretorResult secreteOutsideCellAtBoundaryOnContactWithTotalCount(const Cell &cell, float amount, const std::vector<unsigned char> &types) const { return secreteOutsideCellAtBoundaryOnContactWithTotalCount(cell, field_.c_str(), amount, types); }
    bool secreteInsideCellAtCOM(const Cell &cell, float amount) const { return secreteInsideCellAtCOM(cell, field_.c_str(), amount); }
    FieldSecretorResult secreteInsideCellAtCOMTotalCount(const Cell &cell, float amount) const { return secreteInsideCellAtCOMTotalCount(cell, field_.c_str(), amount); }
    bool uptakeInsideCell(const Cell &cell, float maxUptake, float relativeUptake) const { return uptakeInsideCell(cell, field_.c_str(), maxUptake, relativeUptake); }
    FieldSecretorResult uptakeInsideCellTotalCount(const Cell &cell, float maxUptake, float relativeUptake) const { return uptakeInsideCellTotalCount(cell, field_.c_str(), maxUptake, relativeUptake); }
    bool uptakeInsideCellAtBoundary(const Cell &cell, float maxUptake, float relativeUptake) const { return uptakeInsideCellAtBoundary(cell, field_.c_str(), maxUptake, relativeUptake); }
    FieldSecretorResult uptakeInsideCellAtBoundaryTotalCount(const Cell &cell, float maxUptake, float relativeUptake) const { return uptakeInsideCellAtBoundaryTotalCount(cell, field_.c_str(), maxUptake, relativeUptake); }
    bool uptakeOutsideCellAtBoundary(const Cell &cell, float maxUptake, float relativeUptake) const { return uptakeOutsideCellAtBoundary(cell, field_.c_str(), maxUptake, relativeUptake); }
    FieldSecretorResult uptakeOutsideCellAtBoundaryTotalCount(const Cell &cell, float maxUptake, float relativeUptake) const { return uptakeOutsideCellAtBoundaryTotalCount(cell, field_.c_str(), maxUptake, relativeUptake); }
    float amountSeenByCell(const Cell &cell) const { return amountSeenByCell(cell, field_.c_str()); }
    float totalFieldIntegral() const { return totalFieldIntegral(field_.c_str()); }
    void secreteSingleField(size_t index) const { if (api_ && api_->secrete_single_field_fn) api_->secrete_single_field_fn(api_->userdata, index); }
    void secreteOnContactSingleField(size_t index) const { if (api_ && api_->secrete_contact_single_field_fn) api_->secrete_contact_single_field_fn(api_->userdata, index); }
    void secreteConstantConcentrationSingleField(size_t index) const { if (api_ && api_->secrete_constant_single_field_fn) api_->secrete_constant_single_field_fn(api_->userdata, index); }
private:
    using Call = uint8_t (*)(void *, CC3DCellHandle, const char *, float);
    using TotalCall = CC3DFieldSecretorResultV1 (*)(void *, CC3DCellHandle, const char *, float);
    bool call(Call fn, const Cell &cell, const char *field, float amount) const { return api_ && fn && field ? fn(api_->userdata, cell.handle(), field, amount) != 0 : false; }
    FieldSecretorResult total(TotalCall fn, const Cell &cell, const char *field, float amount) const { return api_ && fn && field ? result(fn(api_->userdata, cell.handle(), field, amount)) : FieldSecretorResult{false, 0}; }
    static FieldSecretorResult result(CC3DFieldSecretorResultV1 value) { return {value.success != 0, value.total_amount}; }
    const CC3DSecretionAPIV1 *api_;
    std::string field_;
};
} // namespace kernel
} // namespace cc3d

inline ::cc3d::kernel::CellIterator CC3DCellsAPIV1::begin() const {
    if (!begin_fn) {
        return ::cc3d::kernel::CellIterator();
    }
    return ::cc3d::kernel::CellIterator(this, begin_fn(userdata));
}

inline ::cc3d::kernel::CellSentinel CC3DCellsAPIV1::end() const {
    return ::cc3d::kernel::CellSentinel();
}

inline ::cc3d::kernel::CellTypes CC3DCellTypesAPIV1::cellTypes() const {
    return ::cc3d::kernel::CellTypes(this);
}

inline ::cc3d::kernel::CellManipulation CC3DCellManipulationAPIV1::cellManipulation() const {
    return ::cc3d::kernel::CellManipulation(this);
}

inline bool CC3DCellTypesAPIV1::tryGet(const char *type_name, unsigned char &type_id) const {
    return cellTypes().tryGet(type_name, type_id);
}

inline int CC3DCellTypesAPIV1::get(const char *type_name) const {
    return cellTypes().get(type_name);
}

inline int CC3DCellTypesAPIV1::operator[](const char *type_name) const {
    return get(type_name);
}

inline std::string CC3DCellTypesAPIV1::name(unsigned char type_id) const {
    return cellTypes().name(type_id);
}

inline std::string CC3DCellTypesAPIV1::name(const ::cc3d::kernel::Cell &cell) const {
    return cellTypes().name(cell);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::byType(unsigned char type_id) const {
    return ::cc3d::kernel::CellTypeRange(this, type_id);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::byType(
    const std::vector<unsigned char> &type_ids
) const {
    return ::cc3d::kernel::CellTypeRange(this, type_ids);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::byType(
    std::initializer_list<unsigned char> type_ids
) const {
    return ::cc3d::kernel::CellTypeRange(this, std::vector<unsigned char>(type_ids));
}

inline ::cc3d::kernel::OptionalCell CC3DCellsAPIV1::fetchCellById(long cell_id) const {
    if (!fetch_by_id_fn) return ::cc3d::kernel::OptionalCell();
    const CC3DCellViewV1 view = fetch_by_id_fn(userdata, cell_id);
    CC3DOptionalCellViewV1 optionalView{};
    optionalView.hasCell = view.handle ? 1u : 0u;
    optionalView.cell = view;
    return ::cc3d::kernel::OptionalCell(optionalView);
}

inline ::cc3d::kernel::OptionalCell CC3DCellsAPIV1::fetch_cell_by_id(long cell_id) const {
    return fetchCellById(cell_id);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::cellListByType(unsigned char type_id) const {
    return byType(type_id);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::cell_list_by_type(unsigned char type_id) const {
    return byType(type_id);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::cellListByType(
    const std::vector<unsigned char> &type_ids
) const {
    return byType(type_ids);
}

inline ::cc3d::kernel::CellTypeRange CC3DCellsAPIV1::cell_list_by_type(
    const std::vector<unsigned char> &type_ids
) const {
    return byType(type_ids);
}

inline ::cc3d::kernel::SharedData CC3DPythonDictionaryAPIV1::sharedData(
    const CC3DPythonDictionaryQueriesAPIV1 *queries) const {
    return ::cc3d::kernel::SharedData(this, queries);
}

namespace cc3d {
namespace kernel {

class XmlElement {
public:
    XmlElement() : api_(nullptr), handle_(nullptr) {}

    XmlElement(const CC3DXMLAPIV1 *api, CC3DXMLHandle handle)
        : api_(api), handle_(handle) {}

    XmlElement(const XmlElement &) = delete;
    XmlElement &operator=(const XmlElement &) = delete;

    XmlElement(XmlElement &&other) noexcept
        : api_(other.api_), handle_(other.handle_) {
        other.api_ = nullptr;
        other.handle_ = nullptr;
    }

    XmlElement &operator=(XmlElement &&other) noexcept {
        if (this != &other) {
            reset();
            api_ = other.api_;
            handle_ = other.handle_;
            other.api_ = nullptr;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~XmlElement() { reset(); }

    bool valid() const { return api_ && handle_; }

    std::string getValue(const char *field) const {
        if (!valid() || !field || !api_->get_value_fn) return {};
        size_t capacity = 256u;
        std::vector<char> buffer(capacity, '\0');
        for (;;) {
            if (api_->get_value_fn(api_->userdata, handle_, field,
                                   buffer.data(), buffer.size())) {
                return std::string(buffer.data());
            }
            capacity *= 2u;
            if (capacity > 1024u * 1024u) return {};
            buffer.assign(capacity, '\0');
        }
    }

    double getValueAsDouble(const char *field) const {
        double value = 0.0;
        if (!valid() || !field || !api_->get_value_as_double_fn ||
            !api_->get_value_as_double_fn(api_->userdata, handle_, field, &value)) {
            return 0.0;
        }
        return value;
    }

    bool setValue(const char *field, const char *value) const {
        return valid() && field && value && api_->set_value_fn &&
               api_->set_value_fn(api_->userdata, handle_, field, value);
    }

    bool setValue(const char *field, double value) const {
        return valid() && field && api_->set_value_as_double_fn &&
               api_->set_value_as_double_fn(api_->userdata, handle_, field, value);
    }

private:
    void reset() {
        if (api_ && handle_ && api_->destroy_element_fn) {
            api_->destroy_element_fn(api_->userdata, handle_);
        }
        api_ = nullptr;
        handle_ = nullptr;
    }

    const CC3DXMLAPIV1 *api_;
    CC3DXMLHandle handle_;
};

}
}

inline ::cc3d::kernel::XmlElement CC3DXMLAPIV1::getElement(const char *element_id) const {
    if (!get_element_fn || !element_id) return ::cc3d::kernel::XmlElement();
    return ::cc3d::kernel::XmlElement(this, get_element_fn(userdata, element_id));
}

inline CC3DGeometryPointV1 CC3DGeometryAPIV1::distanceVector(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const {
    return distance_vector_fn ? distance_vector_fn(userdata, p1, p2) : CC3DGeometryPointV1{0.0, 0.0, 0.0};
}

inline double CC3DGeometryAPIV1::distance(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const {
    return distance_fn ? distance_fn(userdata, p1, p2) : 0.0;
}

inline CC3DGeometryPointV1 CC3DGeometryAPIV1::invariantDistanceVector(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const {
    return invariant_distance_vector_fn ? invariant_distance_vector_fn(userdata, p1, p2) : CC3DGeometryPointV1{0.0, 0.0, 0.0};
}

inline double CC3DGeometryAPIV1::invariantDistance(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const {
    return invariant_distance_fn ? invariant_distance_fn(userdata, p1, p2) : 0.0;
}

inline CC3DGeometryPointV1 CC3DGeometryAPIV1::unconditionalInvariantDistanceVector(CC3DGeometryPointV1 p1, CC3DGeometryPointV1 p2) const {
    return unconditional_invariant_distance_vector_fn ? unconditional_invariant_distance_vector_fn(userdata, p1, p2) : CC3DGeometryPointV1{0.0, 0.0, 0.0};
}

inline double CC3DGeometryAPIV1::vectorNorm(CC3DGeometryPointV1 value) const {
    return vector_norm_fn ? vector_norm_fn(userdata, value) : 0.0;
}

inline CC3DGeometryPointV1 CC3DGeometryAPIV1::distanceVectorBetweenCells(CC3DCellHandle a, CC3DCellHandle b, bool invariant) const {
    return distance_vector_between_cells_fn ? distance_vector_between_cells_fn(userdata, a, b, invariant ? 1u : 0u) : CC3DGeometryPointV1{0.0, 0.0, 0.0};
}

inline double CC3DGeometryAPIV1::distanceBetweenCells(CC3DCellHandle a, CC3DCellHandle b, bool invariant) const {
    return distance_between_cells_fn ? distance_between_cells_fn(userdata, a, b, invariant ? 1u : 0u) : 0.0;
}

inline CC3DGeometryPointV1 CC3DGeometryAPIV1::hexToCartesian(CC3DGeometryPointV1 value) const {
    return hex_to_cartesian_fn ? hex_to_cartesian_fn(userdata, value) : value;
}

inline CC3DGeometryPointV1 CC3DGeometryAPIV1::cartesianToHex(CC3DGeometryPointV1 value) const {
    return cartesian_to_hex_fn ? cartesian_to_hex_fn(userdata, value) : value;
}

inline bool CC3DGeometryAPIV1::isInBounds(CC3DGeometryPointV1 value) const {
    return is_in_bounds_fn && is_in_bounds_fn(userdata, value) != 0u;
}

inline CC3DFieldDimV1 CC3DGeometryAPIV1::fieldDim() const {
    return field_dim_fn ? field_dim_fn(userdata) : CC3DFieldDimV1{0, 0, 0};
}

inline ::cc3d::kernel::CellAttributes CC3DPythonDictionaryAPIV1::forCell(
    const ::cc3d::kernel::Cell &cell,
    const CC3DPythonDictionaryQueriesAPIV1 *queries
) const {
    return ::cc3d::kernel::CellAttributes(this, cell.handle(), queries);
}

inline ::cc3d::kernel::ClusterRange CC3DCellsAPIV1::clusters() const {
    return clusterApi.clusters();
}

inline ::cc3d::kernel::ClusterCellRange CC3DCellsAPIV1::clusterCells(long cluster_id) const {
    return clusterApi.cells(cluster_id);
}

inline ::cc3d::kernel::ClusterRange CC3DClusterCellsAPIV1::clusters() const {
    return ::cc3d::kernel::ClusterRange(this);
}

inline ::cc3d::kernel::ClusterCellRange CC3DClusterCellsAPIV1::cells(long cluster_id) const {
    return ::cc3d::kernel::ClusterCellRange(this, cluster_id);
}

inline ::cc3d::kernel::CellNeighborRange CC3DCellNeighborsAPIV1::forCell(
    const ::cc3d::kernel::Cell &cell
) const {
    return ::cc3d::kernel::CellNeighborRange(this, cell.handle());
}


inline ::cc3d::kernel::PixelRange CC3DCellPixelsAPIV1::forCell(
    const ::cc3d::kernel::Cell &cell
) const {
    return ::cc3d::kernel::PixelRange(this, cell.handle());
}

inline ::cc3d::kernel::ScalarField CC3DScalarFieldsAPIV1::operator[](const char *name) const {
    if (!find_fn || !name) {
        return ::cc3d::kernel::ScalarField();
    }
    return ::cc3d::kernel::ScalarField(this, find_fn(userdata, name));
}

inline ::cc3d::kernel::Dim3 CC3DCellFieldAPIV1::dim() const {
    if (!dim_fn) {
        return ::cc3d::kernel::Dim3{0, 0, 0};
    }
    CC3DFieldDimV1 d = dim_fn(userdata);
    return ::cc3d::kernel::Dim3{d.x, d.y, d.z};
}

inline ::cc3d::kernel::OptionalCell CC3DCellFieldAPIV1::operator()(int x, int y, int z) const {
    if (!get_fn) {
        return ::cc3d::kernel::OptionalCell();
    }
    return ::cc3d::kernel::OptionalCell(get_fn(userdata, x, y, z));
}

inline ::cc3d::kernel::EveryPixelRange CC3DCellFieldAPIV1::everyPixel(
    int step_x, int step_y, int step_z
) const {
    return ::cc3d::kernel::EveryPixelRange(dim(), step_x, step_y, step_z);
}

inline ::cc3d::kernel::EveryPixelRange CC3DCellFieldAPIV1::everyPixelWithSteps(
    int step_x, int step_y, int step_z
) const {
    return everyPixel(step_x, step_y, step_z);
}

inline void CC3DCellFieldAPIV1::assign(
    int x_min,
    int x_max,
    int y_min,
    int y_max,
    int z_min,
    int z_max,
    const ::cc3d::kernel::Cell &cell
) const {
    if (!assign_fn) {
        return;
    }
    assign_fn(userdata, x_min, x_max, y_min, y_max, z_min, z_max, cell.handle());
}

inline ::cc3d::kernel::SBMLModel CC3DSBMLAPIV1::find(
    const ::cc3d::kernel::Cell &cell,
    const char *model_name
) const {
    if (!find_model_fn || !model_name) {
        return ::cc3d::kernel::SBMLModel();
    }
    return ::cc3d::kernel::SBMLModel(this, find_model_fn(userdata, cell.handle(), model_name));
}
inline ::cc3d::kernel::Cell CC3DCellFactoryAPIV1::create(unsigned char cell_type) const {
    if (!create_fn) {
        return ::cc3d::kernel::Cell(CC3DCellViewV1{});
    }
    return ::cc3d::kernel::Cell(create_fn(userdata, cell_type));
}

inline ::cc3d::kernel::AdhesionFlex CC3DAdhesionFlexAPIV1::adhesionFlex() const {
    return ::cc3d::kernel::AdhesionFlex(this);
}

inline ::cc3d::kernel::LengthConstraint CC3DLengthConstraintAPIV1::lengthConstraint() const {
    return ::cc3d::kernel::LengthConstraint(this);
}

inline ::cc3d::kernel::SharedScalarField CC3DSharedScalarFieldsAPIV1::scalar(const char *name) const {
    if (!find_fn || !name) return ::cc3d::kernel::SharedScalarField();
    return ::cc3d::kernel::SharedScalarField(this, find_fn(userdata, name));
}

inline ::cc3d::kernel::SharedScalarField CC3DSharedScalarFieldsAPIV1::createScalar(
    const char *name, CC3DFieldPrecisionV1 precision, int padding
) const {
    if (!create_fn || !name) return ::cc3d::kernel::SharedScalarField();
    return ::cc3d::kernel::SharedScalarField(this, create_fn(userdata, name, precision, padding));
}

inline ::cc3d::kernel::SharedVectorField CC3DSharedVectorFieldsAPIV1::vector(const char *name) const {
    if (!find_fn || !name) return ::cc3d::kernel::SharedVectorField();
    return ::cc3d::kernel::SharedVectorField(this, find_fn(userdata, name));
}

inline ::cc3d::kernel::SharedVectorField CC3DSharedVectorFieldsAPIV1::createVector(
    const char *name, CC3DFieldPrecisionV1 precision, int padding
) const {
    if (!create_fn || !name) return ::cc3d::kernel::SharedVectorField();
    return ::cc3d::kernel::SharedVectorField(this, create_fn(userdata, name, precision, padding));
}

inline ::cc3d::kernel::ClusterSurface CC3DClusterSurfaceAPIV1::clusterSurface() const {
    return ::cc3d::kernel::ClusterSurface(this);
}

inline ::cc3d::kernel::ClusterSurfaceTracker CC3DClusterSurfaceTrackerAPIV1::clusterSurfaceTracker() const {
    return ::cc3d::kernel::ClusterSurfaceTracker(this);
}

inline ::cc3d::kernel::Mitosis CC3DMitosisAPIV1::mitosis() const {
    return ::cc3d::kernel::Mitosis(this);
}

inline ::cc3d::kernel::OrientedGrowth CC3DOrientedGrowthAPIV1::orientedGrowth() const {
    return ::cc3d::kernel::OrientedGrowth(this);
}

inline ::cc3d::kernel::MomentOfInertia CC3DMomentOfInertiaAPIV1::momentOfInertia() const {
    return ::cc3d::kernel::MomentOfInertia(this);
}

inline ::cc3d::kernel::FocalPointPlasticity CC3DFocalPointPlasticityAPIV1::focalPointPlasticity() const {
    return ::cc3d::kernel::FocalPointPlasticity(this);
}
inline ::cc3d::kernel::FocalPointPlasticity
CC3DFocalPointPlasticityQueriesAPIV1::focalPointPlasticity(
    const CC3DFocalPointPlasticityAPIV1 *base) const {
    return ::cc3d::kernel::FocalPointPlasticity(base, this);
}
inline ::cc3d::kernel::PolarizationVector CC3DPolarizationVectorAPIV1::polarizationVector() const {
    return ::cc3d::kernel::PolarizationVector(this);
}
inline ::cc3d::kernel::CellOrientation CC3DCellOrientationAPIV1::cellOrientation() const {
    return ::cc3d::kernel::CellOrientation(this);
}
inline ::cc3d::kernel::Chemotaxis CC3DChemotaxisAPIV1::chemotaxis() const {
    return ::cc3d::kernel::Chemotaxis(this);
}
inline ::cc3d::kernel::FieldSecretor CC3DSecretionAPIV1::fieldSecretor() const {
    return ::cc3d::kernel::FieldSecretor(this);
}

inline ::cc3d::kernel::FieldSecretor CC3DSecretionAPIV1::fieldSecretor(const char *field) const {
    return ::cc3d::kernel::FieldSecretor(this, field);
}

inline ::cc3d::kernel::Cell CC3DKernelContext::newCell(unsigned char cell_type) const {
    return cellFactory.create(cell_type);
}
#endif

#endif
