#include "AdhesionFlexPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>
#include <vector>

namespace {
CompuCell3D::AdhesionFlexPlugin *p(void *x) { return static_cast<CompuCell3D::AdhesionFlexPlugin *>(x); }
size_t count(void *x) { return p(x)->getAdhesionMoleculeNameVec().size(); }
const char *name(void *x, size_t i) { static thread_local std::vector<std::string> n; n=p(x)->getAdhesionMoleculeNameVec(); return i<n.size()?n[i].c_str():nullptr; }
float gd(void *x, CC3DPluginCellHandle c, const char *n) { return p(x)->getAdhesionMoleculeDensity(static_cast<CompuCell3D::CellG *>(c), n); }
float gdi(void *x, CC3DPluginCellHandle c, size_t i) { return p(x)->getAdhesionMoleculeDensityByIndex(static_cast<CompuCell3D::CellG *>(c), static_cast<int>(i)); }
void sd(void *x, CC3DPluginCellHandle c, const char *n, float v) { p(x)->setAdhesionMoleculeDensity(static_cast<CompuCell3D::CellG *>(c), n, v); }
void sdi(void *x, CC3DPluginCellHandle c, size_t i, float v) { p(x)->setAdhesionMoleculeDensityByIndex(static_cast<CompuCell3D::CellG *>(c), static_cast<int>(i), v); }
uint8_t sdv(void *x, CC3DPluginCellHandle c, const float *v, size_t n) { p(x)->setAdhesionMoleculeDensityVector(static_cast<CompuCell3D::CellG *>(c), std::vector<float>(v,v+n)); return 1; }
void adv(void *x, CC3DPluginCellHandle c, const float *v, size_t n) { p(x)->assignNewAdhesionMoleculeDensityVector(static_cast<CompuCell3D::CellG *>(c), std::vector<float>(v,v+n)); }
float gmd(void *x, const char *n) { return p(x)->getMediumAdhesionMoleculeDensity(n); }
float gmdi(void *x, size_t i) { return p(x)->getMediumAdhesionMoleculeDensityByIndex(static_cast<int>(i)); }
void smd(void *x, const char *n, float v) { p(x)->setMediumAdhesionMoleculeDensity(n,v); }
void smdi(void *x, size_t i, float v) { p(x)->setMediumAdhesionMoleculeDensityByIndex(static_cast<int>(i),v); }
uint8_t smdv(void *x, const float *v, size_t n) { p(x)->setMediumAdhesionMoleculeDensityVector(std::vector<float>(v,v+n)); return 1; }
void amdv(void *x, const float *v, size_t n) { p(x)->assignNewMediumAdhesionMoleculeDensityVector(std::vector<float>(v,v+n)); }
}
extern "C" void cc3d_register_adhesion_flex_plugin(void *simulator, void *plugin) {
    CC3DAdhesionFlexCallbacksV1 c{}; c.plugin_instance=plugin; c.molecule_count_fn=count; c.molecule_name_fn=name; c.get_density_fn=gd; c.get_density_by_index_fn=gdi; c.set_density_fn=sd; c.set_density_by_index_fn=sdi; c.set_density_vector_fn=sdv; c.assign_density_vector_fn=adv; c.get_medium_density_fn=gmd; c.get_medium_density_by_index_fn=gmdi; c.set_medium_density_fn=smd; c.set_medium_density_by_index_fn=smdi; c.set_medium_density_vector_fn=smdv; c.assign_medium_density_vector_fn=amdv; cc3d_register_adhesion_flex_callbacks(simulator,&c);
}
