#if defined(_WIN32)
  #ifdef StreamRedirectorsShared_EXPORTS
  #define STREAMREDIRECTORS_EXPORT __declspec(dllexport)
  #define STREAMREDIRECTORS_EXPIMP_TEMPLATE
  #else
  #define STREAMREDIRECTORS_EXPORT __declspec(dllimport)
  #define CSTREAMREDIRECTORS_EXPIMP_TEMPLATE extern
  #endif
#else
     #define STREAMREDIRECTORS_EXPORT
     #define STREAMREDIRECTORS_EXPIMP_TEMPLATE
#endif


