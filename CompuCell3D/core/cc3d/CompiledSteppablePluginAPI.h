#ifndef CC3D_COMPILED_STEPPABLE_PLUGIN_API_H
#define CC3D_COMPILED_STEPPABLE_PLUGIN_API_H
#include <stddef.h>
#include <stdint.h>
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
} CC3DFocalPointPlasticityCallbacksV1;

typedef struct CC3DPolarizationVectorCallbacksV1 {
    void *plugin_instance;
    void (*set_fn)(void *, CC3DPluginCellHandle, float, float, float);
    uint8_t (*get_fn)(void *, CC3DPluginCellHandle, float *, float *, float *);
} CC3DPolarizationVectorCallbacksV1;

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
CC3D_PLUGIN_API_EXPORT void cc3d_register_adhesion_flex_callbacks(CC3DPluginSimulatorHandle, const CC3DAdhesionFlexCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DAdhesionFlexCallbacksV1 *cc3d_get_adhesion_flex_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_adhesion_flex_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_focal_point_plasticity_callbacks(CC3DPluginSimulatorHandle, const CC3DFocalPointPlasticityCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DFocalPointPlasticityCallbacksV1 *cc3d_get_focal_point_plasticity_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_focal_point_plasticity_callbacks(void *plugin_instance);
CC3D_PLUGIN_API_EXPORT void cc3d_register_polarization_vector_callbacks(CC3DPluginSimulatorHandle, const CC3DPolarizationVectorCallbacksV1 *);
CC3D_PLUGIN_API_EXPORT const CC3DPolarizationVectorCallbacksV1 *cc3d_get_polarization_vector_callbacks(CC3DPluginSimulatorHandle);
CC3D_PLUGIN_API_EXPORT void cc3d_unregister_polarization_vector_callbacks(void *plugin_instance);
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
