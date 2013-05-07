#ifndef rrCExporterH
#define rrCExporterH

#if defined(WIN32)

#if defined(BUILD_MODEL_DLL)
#define D_S __declspec(dllexport)
#else
#define D_S __declspec(dllimport)
#endif

#else
#define D_S 
#endif
#endif


