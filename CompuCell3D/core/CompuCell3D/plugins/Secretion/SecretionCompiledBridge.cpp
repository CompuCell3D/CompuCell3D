#include "SecretionPlugin.h"
#include "FieldSecretor.h"
#include <cc3d/CompiledSteppablePluginAPI.h>

namespace {
using Plugin = CompuCell3D::SecretionPlugin;
using Cell = CompuCell3D::CellG;
using Secretor = CompuCell3D::FieldSecretor;

Secretor secretor(void *p, const char *field) { return static_cast<Plugin *>(p)->getFieldSecretor(field ? field : ""); }
CC3DFieldSecretorResultV1 result(const CompuCell3D::FieldSecretorResult &v) { return {static_cast<uint8_t>(v.success_flag), v.tot_amount}; }
std::vector<unsigned char> types(const unsigned char *v, size_t n) { return v ? std::vector<unsigned char>(v, v + n) : std::vector<unsigned char>(); }
#define BOOL(name, method) uint8_t name(void *p, CC3DPluginCellHandle c, const char *f, float a) { return secretor(p, f).method(static_cast<Cell *>(c), a) ? 1u : 0u; }
#define TOTAL(name, method) CC3DFieldSecretorResultV1 name(void *p, CC3DPluginCellHandle c, const char *f, float a) { return result(secretor(p, f).method(static_cast<Cell *>(c), a)); }
BOOL(secrete_inside, _secreteInsideCell)
TOTAL(secrete_inside_total, _secreteInsideCellTotalCount)
BOOL(secrete_inside_constant, _secreteInsideCellConstantConcentration)
TOTAL(secrete_inside_constant_total, _secreteInsideCellConstantConcentrationTotalCount)
BOOL(secrete_inside_boundary, _secreteInsideCellAtBoundary)
TOTAL(secrete_inside_boundary_total, _secreteInsideCellAtBoundaryTotalCount)
BOOL(secrete_outside_boundary, _secreteOutsideCellAtBoundary)
TOTAL(secrete_outside_boundary_total, _secreteOutsideCellAtBoundaryTotalCount)
BOOL(secrete_com, secreteInsideCellAtCOM)
TOTAL(secrete_com_total, secreteInsideCellAtCOMTotalCount)

uint8_t secrete_inside_boundary_contact(void *p, CC3DPluginCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { return secretor(p, f)._secreteInsideCellAtBoundaryOnContactWith(static_cast<Cell *>(c), a, types(t, n)) ? 1u : 0u; }
CC3DFieldSecretorResultV1 secrete_inside_boundary_contact_total(void *p, CC3DPluginCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { return result(secretor(p, f)._secreteInsideCellAtBoundaryOnContactWithTotalCount(static_cast<Cell *>(c), a, types(t, n))); }
uint8_t secrete_outside_boundary_contact(void *p, CC3DPluginCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { return secretor(p, f)._secreteOutsideCellAtBoundaryOnContactWith(static_cast<Cell *>(c), a, types(t, n)) ? 1u : 0u; }
CC3DFieldSecretorResultV1 secrete_outside_boundary_contact_total(void *p, CC3DPluginCellHandle c, const char *f, float a, const unsigned char *t, size_t n) { return result(secretor(p, f)._secreteOutsideCellAtBoundaryOnContactWithTotalCount(static_cast<Cell *>(c), a, types(t, n))); }

uint8_t uptake_inside(void *p, CC3DPluginCellHandle c, const char *f, float max, float rel) { return secretor(p, f)._uptakeInsideCell(static_cast<Cell *>(c), max, rel) ? 1u : 0u; }
CC3DFieldSecretorResultV1 uptake_inside_total(void *p, CC3DPluginCellHandle c, const char *f, float max, float rel) { return result(secretor(p, f)._uptakeInsideCellTotalCount(static_cast<Cell *>(c), max, rel)); }
uint8_t uptake_boundary(void *p, CC3DPluginCellHandle c, const char *f, float max, float rel) { return secretor(p, f)._uptakeInsideCellAtBoundary(static_cast<Cell *>(c), max, rel) ? 1u : 0u; }
CC3DFieldSecretorResultV1 uptake_boundary_total(void *p, CC3DPluginCellHandle c, const char *f, float max, float rel) { return result(secretor(p, f)._uptakeInsideCellAtBoundaryTotalCount(static_cast<Cell *>(c), max, rel)); }
uint8_t uptake_outside_boundary(void *p, CC3DPluginCellHandle c, const char *f, float max, float rel) { return secretor(p, f)._uptakeOutsideCellAtBoundary(static_cast<Cell *>(c), max, rel) ? 1u : 0u; }
CC3DFieldSecretorResultV1 uptake_outside_boundary_total(void *p, CC3DPluginCellHandle c, const char *f, float max, float rel) { return result(secretor(p, f)._uptakeOutsideCellAtBoundaryTotalCount(static_cast<Cell *>(c), max, rel)); }
float amount_seen(void *p, CC3DPluginCellHandle c, const char *f) { return secretor(p, f)._amountSeenByCell(static_cast<Cell *>(c)); }
float field_integral(void *p, const char *f) { return secretor(p, f).totalFieldIntegral(); }
void single(void *p, size_t i) { static_cast<Plugin *>(p)->secreteSingleField(static_cast<unsigned int>(i)); }
void contact_single(void *p, size_t i) { static_cast<Plugin *>(p)->secreteOnContactSingleField(static_cast<unsigned int>(i)); }
void constant_single(void *p, size_t i) { static_cast<Plugin *>(p)->secreteConstantConcentrationSingleField(static_cast<unsigned int>(i)); }
#undef BOOL
#undef TOTAL
}

extern "C" void cc3d_register_secretion_plugin(void *simulator, void *plugin) {
    CC3DSecretionCallbacksV1 c{};
    c.plugin_instance = plugin;
    c.secrete_inside_fn = &secrete_inside; c.secrete_inside_total_fn = &secrete_inside_total;
    c.secrete_inside_constant_fn = &secrete_inside_constant; c.secrete_inside_constant_total_fn = &secrete_inside_constant_total;
    c.secrete_inside_boundary_fn = &secrete_inside_boundary; c.secrete_inside_boundary_total_fn = &secrete_inside_boundary_total;
    c.secrete_inside_boundary_contact_fn = &secrete_inside_boundary_contact; c.secrete_inside_boundary_contact_total_fn = &secrete_inside_boundary_contact_total;
    c.secrete_outside_boundary_fn = &secrete_outside_boundary; c.secrete_outside_boundary_total_fn = &secrete_outside_boundary_total;
    c.secrete_outside_boundary_contact_fn = &secrete_outside_boundary_contact; c.secrete_outside_boundary_contact_total_fn = &secrete_outside_boundary_contact_total;
    c.secrete_com_fn = &secrete_com; c.secrete_com_total_fn = &secrete_com_total;
    c.uptake_inside_fn = &uptake_inside; c.uptake_inside_total_fn = &uptake_inside_total;
    c.uptake_boundary_fn = &uptake_boundary; c.uptake_boundary_total_fn = &uptake_boundary_total;
    c.uptake_outside_boundary_fn = &uptake_outside_boundary; c.uptake_outside_boundary_total_fn = &uptake_outside_boundary_total;
    c.amount_seen_fn = &amount_seen; c.field_integral_fn = &field_integral;
    c.secrete_single_field_fn = &single; c.secrete_contact_single_field_fn = &contact_single; c.secrete_constant_single_field_fn = &constant_single;
    cc3d_register_secretion_callbacks(simulator, &c);
}
