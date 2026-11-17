#ifndef CC3D_COMPILED_STEPPABLE_PLUGIN_API_H
#define CC3D_COMPILED_STEPPABLE_PLUGIN_API_H
#include <stddef.h>
#include <stdint.h>
#include "kernel.h"
#if defined(_WIN32)
# if defined(CompuCellLibShared_EXPORTS)
#  define CC3D_PLUGIN_API_EXPORT __declspec(dllexport)
# else
#  define CC3D_PLUGIN_API_EXPORT __declspec(dllimport)
# endif
#else
# define CC3D_PLUGIN_API_EXPORT
#endif
typedef void *CC3DPluginSimulatorHandle;
typedef void *CC3DPluginCellHandle;
typedef struct CC3DLengthConstraintCallbacksV1 {
    void *plugin_instance;
    void (*set_fn)(void *, CC3DPluginCellHandle, double, double, double);
    double (*get_lambda_fn)(void *, CC3DPluginCellHandle);
    double (*get_target_fn)(void *, CC3DPluginCellHandle);
    double (*get_minor_target_fn)(void *, CC3DPluginCellHandle);
} CC3DLengthConstraintCallbacksV1;

typedef struct CC3DClusterSurfaceCallbacksV1 {
    void *plugin_instance;
    void (*set_fn)(void *, CC3DPluginCellHandle, float, float);
    void (*get_fn)(void *, CC3DPluginCellHandle, float *, float *);
} CC3DClusterSurfaceCallbacksV1;

typedef struct CC3DClusterSurfaceTrackerCallbacksV1 {
    void *plugin_instance;
    void (*update_cluster_surface_fn)(void *, long);
} CC3DClusterSurfaceTrackerCallbacksV1;

typedef struct CC3DMitosisCallbacksV1 {
    void *steppable_instance;
    void (*set_parent_child_position_fn)(void *, int);
    int (*get_parent_child_position_fn)(void *);
    uint8_t (*divide_random_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*divide_orientation_fn)(void *, CC3DPluginCellHandle, double, double, double);
    uint8_t (*divide_major_axis_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*divide_minor_axis_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*divide_cluster_random_fn)(void *, long);
    uint8_t (*divide_cluster_orientation_fn)(void *, long, double, double, double);
    uint8_t (*divide_cluster_major_axis_fn)(void *, long);
    uint8_t (*divide_cluster_minor_axis_fn)(void *, long);
    CC3DPluginCellHandle (*parent_cell_fn)(void *);
    CC3DPluginCellHandle (*child_cell_fn)(void *);
} CC3DMitosisCallbacksV1;

typedef struct CC3DOrientedGrowthCallbacksV1 {
    void *plugin_instance;
    void (*set_constraint_width_fn)(void *, CC3DPluginCellHandle, float);
    void (*set_elongation_axis_fn)(void *, CC3DPluginCellHandle, float, float);
    void (*set_elongation_enabled_fn)(void *, CC3DPluginCellHandle, uint8_t);
    float (*get_elongation_axis_x_fn)(void *, CC3DPluginCellHandle);
    float (*get_elongation_axis_y_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*get_elongation_enabled_fn)(void *, CC3DPluginCellHandle);
} CC3DOrientedGrowthCallbacksV1;

typedef struct CC3DMomentOfInertiaCallbacksV1 {
    void *plugin_instance;
    size_t (*get_semiaxes_fn)(void *, CC3DPluginCellHandle, double *, size_t);
    size_t (*get_semiaxes_xy_fn)(void *, CC3DPluginCellHandle, double *, size_t);
    size_t (*get_semiaxes_xz_fn)(void *, CC3DPluginCellHandle, double *, size_t);
    size_t (*get_semiaxes_yz_fn)(void *, CC3DPluginCellHandle, double *, size_t);
    size_t (*get_semiaxes_3d_fn)(void *, CC3DPluginCellHandle, double *, size_t);
} CC3DMomentOfInertiaCallbacksV1;

typedef struct CC3DAdhesionFlexCallbacksV1 {
    void *plugin_instance;
    size_t (*molecule_count_fn)(void *);
    const char *(*molecule_name_fn)(void *, size_t);
    float (*get_density_fn)(void *, CC3DPluginCellHandle, const char *);
    float (*get_density_by_index_fn)(void *, CC3DPluginCellHandle, size_t);
    void (*set_density_fn)(void *, CC3DPluginCellHandle, const char *, float);
    void (*set_density_by_index_fn)(void *, CC3DPluginCellHandle, size_t, float);
    uint8_t (*set_density_vector_fn)(void *, CC3DPluginCellHandle, const float *, size_t);
    void (*assign_density_vector_fn)(void *, CC3DPluginCellHandle, const float *, size_t);
    float (*get_medium_density_fn)(void *, const char *);
    float (*get_medium_density_by_index_fn)(void *, size_t);
    void (*set_medium_density_fn)(void *, const char *, float);
    void (*set_medium_density_by_index_fn)(void *, size_t, float);
    uint8_t (*set_medium_density_vector_fn)(void *, const float *, size_t);
    void (*assign_medium_density_vector_fn)(void *, const float *, size_t);
} CC3DAdhesionFlexCallbacksV1;

typedef struct CC3DFocalPointPlasticityCallbacksV1 {
    void *plugin_instance;
    void (*create_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle, double, double, double);
    void (*create_internal_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle, double, double, double);
    void (*delete_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle);
    void (*delete_internal_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle);
    void (*set_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle, double, double, double);
    double (*get_lambda_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle);
    double (*get_target_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle);
    int (*create_anchor_fn)(void *, CC3DPluginCellHandle, double, double, double, float, float, float);
    void (*delete_anchor_fn)(void *, CC3DPluginCellHandle, int);
    void (*set_anchor_fn)(void *, CC3DPluginCellHandle, int, double, double, double, float, float, float);
    CC3DFPPLinkIteratorHandle (*links_begin_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*links_valid_fn)(void *, CC3DFPPLinkIteratorHandle);
    CC3DFPPLinkViewV1 (*links_get_fn)(void *, CC3DFPPLinkIteratorHandle);
    void (*links_next_fn)(void *, CC3DFPPLinkIteratorHandle);
    void (*links_destroy_fn)(void *, CC3DFPPLinkIteratorHandle);
    CC3DFPPLinkIteratorHandle (*internal_links_begin_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*internal_links_valid_fn)(void *, CC3DFPPLinkIteratorHandle);
    CC3DFPPLinkViewV1 (*internal_links_get_fn)(void *, CC3DFPPLinkIteratorHandle);
    void (*internal_links_next_fn)(void *, CC3DFPPLinkIteratorHandle);
    void (*internal_links_destroy_fn)(void *, CC3DFPPLinkIteratorHandle);
    CC3DFPPLinkIteratorHandle (*anchors_begin_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*anchors_valid_fn)(void *, CC3DFPPLinkIteratorHandle);
    CC3DFPPLinkViewV1 (*anchors_get_fn)(void *, CC3DFPPLinkIteratorHandle);
    void (*anchors_next_fn)(void *, CC3DFPPLinkIteratorHandle);
    void (*anchors_destroy_fn)(void *, CC3DFPPLinkIteratorHandle);
    CC3DFPPLinkViewV1 (*find_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle);
    CC3DFPPLinkViewV1 (*find_internal_link_fn)(void *, CC3DPluginCellHandle, CC3DPluginCellHandle);
    size_t (*link_count_fn)(void *, CC3DPluginCellHandle);
    size_t (*internal_link_count_fn)(void *, CC3DPluginCellHandle);
    size_t (*anchor_count_fn)(void *, CC3DPluginCellHandle);
    CC3DFPPCellIteratorHandle (*linked_cells_begin_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*linked_cells_valid_fn)(void *, CC3DFPPCellIteratorHandle);
    CC3DCellViewV1 (*linked_cells_get_fn)(void *, CC3DFPPCellIteratorHandle);
    void (*linked_cells_next_fn)(void *, CC3DFPPCellIteratorHandle);
    void (*linked_cells_destroy_fn)(void *, CC3DFPPCellIteratorHandle);
    CC3DFPPCellIteratorHandle (*internal_linked_cells_begin_fn)(void *, CC3DPluginCellHandle);
    uint8_t (*internal_linked_cells_valid_fn)(void *, CC3DFPPCellIteratorHandle);
    CC3DCellViewV1 (*internal_linked_cells_get_fn)(void *, CC3DFPPCellIteratorHandle);
    void (*internal_linked_cells_next_fn)(void *, CC3DFPPCellIteratorHandle);
    void (*internal_linked_cells_destroy_fn)(void *, CC3DFPPCellIteratorHandle);
} CC3DFocalPointPlasticityCallbacksV1;

typedef struct CC3DPolarizationVectorCallbacksV1 {
    void *plugin_instance;
    void (*set_fn)(void *, CC3DPluginCellHandle, float, float, float);
    uint8_t (*get_fn)(void *, CC3DPluginCellHandle, float *, float *, float *);
} CC3DPolarizationVectorCallbacksV1;

typedef struct CC3DCellOrientationCallbacksV1 {
    void *plugin_instance;
    void (*set_lambda_fn)(void *, CC3DPluginCellHandle, double);
    double (*get_lambda_fn)(void *, CC3DPluginCellHandle);
} CC3DCellOrientationCallbacksV1;

typedef struct CC3DChemotaxisCallbacksV1 {
    void *plugin_instance;
    size_t (*field_count_fn)(void *, CC3DPluginCellHandle);
    const char *(*field_name_fn)(void *, CC3DPluginCellHandle, size_t);
    void *(*add_data_fn)(void *, CC3DPluginCellHandle, const char *);
    void *(*get_data_fn)(void *, CC3DPluginCellHandle, const char *);
    float (*data_get_lambda_fn)(void *);
    void (*data_set_lambda_fn)(void *, float);
    float (*data_get_saturation_fn)(void *);
    void (*data_set_saturation_fn)(void *, float);
    const char *(*data_get_formula_name_fn)(void *);
    uint8_t (*data_set_formula_name_fn)(void *, const char *);
    size_t (*data_get_toward_count_fn)(void *);
    int (*data_get_toward_type_fn)(void *, size_t);
    void (*data_assign_toward_types_fn)(void *, const int *, size_t);
} CC3DChemotaxisCallbacksV1;

#ifndef CC3D_FIELD_SECRETOR_RESULT_V1_DEFINED
#define CC3D_FIELD_SECRETOR_RESULT_V1_DEFINED
typedef struct CC3DFieldSecretorResultV1 {
    uint8_t success;
    float total_amount;
} CC3DFieldSecretorResultV1;
#endif

typedef struct CC3DSecretionCallbacksV1 {
    void *plugin_instance;
    uint8_t (*secrete_inside_fn)(void *, CC3DPluginCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_inside_total_fn)(void *, CC3DPluginCellHandle, const char *, float);
    uint8_t (*secrete_inside_constant_fn)(void *, CC3DPluginCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_inside_constant_total_fn)(void *, CC3DPluginCellHandle, const char *, float);
    uint8_t (*secrete_inside_boundary_fn)(void *, CC3DPluginCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_inside_boundary_total_fn)(void *, CC3DPluginCellHandle, const char *, float);
    uint8_t (*secrete_inside_boundary_contact_fn)(void *, CC3DPluginCellHandle, const char *, float, const unsigned char *, size_t);
    CC3DFieldSecretorResultV1 (*secrete_inside_boundary_contact_total_fn)(void *, CC3DPluginCellHandle, const char *, float, const unsigned char *, size_t);
    uint8_t (*secrete_outside_boundary_fn)(void *, CC3DPluginCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_outside_boundary_total_fn)(void *, CC3DPluginCellHandle, const char *, float);
    uint8_t (*secrete_outside_boundary_contact_fn)(void *, CC3DPluginCellHandle, const char *, float, const unsigned char *, size_t);
    CC3DFieldSecretorResultV1 (*secrete_outside_boundary_contact_total_fn)(void *, CC3DPluginCellHandle, const char *, float, const unsigned char *, size_t);
    uint8_t (*secrete_com_fn)(void *, CC3DPluginCellHandle, const char *, float);
    CC3DFieldSecretorResultV1 (*secrete_com_total_fn)(void *, CC3DPluginCellHandle, const char *, float);
    uint8_t (*uptake_inside_fn)(void *, CC3DPluginCellHandle, const char *, float, float);
    CC3DFieldSecretorResultV1 (*uptake_inside_total_fn)(void *, CC3DPluginCellHandle, const char *, float, float);
    uint8_t (*uptake_boundary_fn)(void *, CC3DPluginCellHandle, const char *, float, float);
    CC3DFieldSecretorResultV1 (*uptake_boundary_total_fn)(void *, CC3DPluginCellHandle, const char *, float, float);
    uint8_t (*uptake_outside_boundary_fn)(void *, CC3DPluginCellHandle, const char *, float, float);
    CC3DFieldSecretorResultV1 (*uptake_outside_boundary_total_fn)(void *, CC3DPluginCellHandle, const char *, float, float);
    float (*amount_seen_fn)(void *, CC3DPluginCellHandle, const char *);
    float (*field_integral_fn)(void *, const char *);
    void (*secrete_single_field_fn)(void *, size_t);
    void (*secrete_contact_single_field_fn)(void *, size_t);
    void (*secrete_constant_single_field_fn)(void *, size_t);
} CC3DSecretionCallbacksV1;
#ifdef __cplusplus
extern "C" {
#endif
CC3D_PLUGIN_API_EXPORT void cc3d_register_length_constraint_callbacks(CC3DPluginSimulatorHandle, const CC3DLengthConstraintCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DLengthConstraintCallbacksV1 *cc3d_get_length_constraint_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_length_constraint_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_cluster_surface_callbacks(CC3DPluginSimulatorHandle, const CC3DClusterSurfaceCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DClusterSurfaceCallbacksV1 *cc3d_get_cluster_surface_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_cluster_surface_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_cluster_surface_tracker_callbacks(CC3DPluginSimulatorHandle, const CC3DClusterSurfaceTrackerCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DClusterSurfaceTrackerCallbacksV1 *cc3d_get_cluster_surface_tracker_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_cluster_surface_tracker_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_mitosis_callbacks(CC3DPluginSimulatorHandle, const CC3DMitosisCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DMitosisCallbacksV1 *cc3d_get_mitosis_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_mitosis_callbacks(void *steppable_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_oriented_growth_callbacks(CC3DPluginSimulatorHandle, const CC3DOrientedGrowthCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DOrientedGrowthCallbacksV1 *cc3d_get_oriented_growth_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_oriented_growth_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_moment_of_inertia_callbacks(CC3DPluginSimulatorHandle, const CC3DMomentOfInertiaCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DMomentOfInertiaCallbacksV1 *cc3d_get_moment_of_inertia_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_moment_of_inertia_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_adhesion_flex_callbacks(CC3DPluginSimulatorHandle, const CC3DAdhesionFlexCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DAdhesionFlexCallbacksV1 *cc3d_get_adhesion_flex_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_adhesion_flex_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_focal_point_plasticity_callbacks(CC3DPluginSimulatorHandle, const CC3DFocalPointPlasticityCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DFocalPointPlasticityCallbacksV1 *cc3d_get_focal_point_plasticity_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_focal_point_plasticity_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_polarization_vector_callbacks(CC3DPluginSimulatorHandle, const CC3DPolarizationVectorCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DPolarizationVectorCallbacksV1 *cc3d_get_polarization_vector_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_polarization_vector_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_cell_orientation_callbacks(CC3DPluginSimulatorHandle, const CC3DCellOrientationCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DCellOrientationCallbacksV1 *cc3d_get_cell_orientation_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_cell_orientation_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_chemotaxis_callbacks(CC3DPluginSimulatorHandle, const CC3DChemotaxisCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DChemotaxisCallbacksV1 *cc3d_get_chemotaxis_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_chemotaxis_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_secretion_callbacks(CC3DPluginSimulatorHandle, const CC3DSecretionCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DSecretionCallbacksV1 *cc3d_get_secretion_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_secretion_callbacks(void *plugin_instance);
#ifdef __cplusplus
}
#endif
#endif
