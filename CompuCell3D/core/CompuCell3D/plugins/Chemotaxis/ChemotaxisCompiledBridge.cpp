#include "ChemotaxisPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
using Plugin = CompuCell3D::ChemotaxisPlugin;
using Data = CompuCell3D::ChemotaxisData;

size_t field_count(void *p, CC3DPluginCellHandle cell) { return static_cast<Plugin *>(p)->getFieldNamesWithChemotaxisData(static_cast<CompuCell3D::CellG *>(cell)).size(); }
const char *field_name(void *p, CC3DPluginCellHandle cell, size_t index) {
    static thread_local std::vector<std::string> names;
    names = static_cast<Plugin *>(p)->getFieldNamesWithChemotaxisData(static_cast<CompuCell3D::CellG *>(cell));
    return index < names.size() ? names[index].c_str() : nullptr;
}
void *add_data(void *p, CC3DPluginCellHandle cell, const char *name) { return name ? static_cast<void *>(static_cast<Plugin *>(p)->addChemotaxisData(static_cast<CompuCell3D::CellG *>(cell), name)) : nullptr; }
void *get_data(void *p, CC3DPluginCellHandle cell, const char *name) { return name ? static_cast<void *>(static_cast<Plugin *>(p)->getChemotaxisData(static_cast<CompuCell3D::CellG *>(cell), name)) : nullptr; }
float get_lambda(void *p) { return p ? static_cast<Data *>(p)->getLambda() : 0.0f; }
void set_lambda(void *p, float v) { if (p) static_cast<Data *>(p)->setLambda(v); }
float get_saturation(void *p) { return p ? static_cast<Data *>(p)->saturationCoef : 0.0f; }
void set_saturation(void *p, float v) { if (p) static_cast<Data *>(p)->saturationCoef = v; }
const char *get_formula(void *p) { return p ? static_cast<Data *>(p)->formulaName.c_str() : nullptr; }
uint8_t set_formula(void *p, const char *name) { if (!p || !name) return 0; static_cast<Data *>(p)->setChemotaxisFormulaByName(name); return 1; }
size_t toward_count(void *p) { return p ? static_cast<Data *>(p)->chemotactTowardsTypesVec.size() : 0; }
int toward_type(void *p, size_t i) { return p && i < static_cast<Data *>(p)->chemotactTowardsTypesVec.size() ? static_cast<Data *>(p)->chemotactTowardsTypesVec[i] : -1; }
void assign_toward(void *p, const int *types, size_t count) { if (p && (count == 0 || types)) static_cast<Data *>(p)->assignChemotactTowardsVectorTypes(std::vector<int>(types, types + count)); }
}

extern "C" void cc3d_register_chemotaxis_plugin(void *simulator, void *plugin) {
    CC3DChemotaxisCallbacksV1 callbacks{};
    callbacks.plugin_instance = plugin;
    callbacks.field_count_fn = &field_count;
    callbacks.field_name_fn = &field_name;
    callbacks.add_data_fn = &add_data;
    callbacks.get_data_fn = &get_data;
    callbacks.data_get_lambda_fn = &get_lambda;
    callbacks.data_set_lambda_fn = &set_lambda;
    callbacks.data_get_saturation_fn = &get_saturation;
    callbacks.data_set_saturation_fn = &set_saturation;
    callbacks.data_get_formula_name_fn = &get_formula;
    callbacks.data_set_formula_name_fn = &set_formula;
    callbacks.data_get_toward_count_fn = &toward_count;
    callbacks.data_get_toward_type_fn = &toward_type;
    callbacks.data_assign_toward_types_fn = &assign_toward;
    cc3d_register_chemotaxis_callbacks(simulator, &callbacks);
}
