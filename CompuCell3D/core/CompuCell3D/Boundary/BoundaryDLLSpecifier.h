#if defined(_WIN32)
  #ifdef BoundaryShared_EXPORTS
  #define BOUNDARYSHARED_EXPORT __declspec(dllexport)
  #define BOUNDARYSHARED_EXPIMP_TEMPLATE
  #else
    #define BOUNDARYSHARED_EXPORT __declspec(dllimport)
    #define BOUNDARYSHARED_EXPIMP_TEMPLATE extern
  #endif
#else
     #define BOUNDARYSHARED_EXPORT
     #define BOUNDARYSHARED_EXPIMP_TEMPLATE
#endif


