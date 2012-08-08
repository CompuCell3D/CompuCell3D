#if defined(_WIN32)
  #ifdef CompuCellLibShared_EXPORTS
  #define COMPUCELLLIB_EXPORT __declspec(dllexport)
  #define COMPUCELLLIB_EXPIMP_TEMPLATE
  #else
  #define COMPUCELLLIB_EXPORT __declspec(dllimport)
  #define COMPUCELLLIB_EXPIMP_TEMPLATE extern
  #endif
#else
     #define COMPUCELLLIB_EXPORT
     #define COMPUCELLLIB_EXPIMP_TEMPLATE
#endif


