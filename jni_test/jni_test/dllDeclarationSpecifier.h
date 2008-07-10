#if defined(_WIN32)
  #ifdef EXP_STL
  #define DECLSPECIFIER __declspec(dllexport)
  #define EXPIMP_TEMPLATE
  #else
  #define DECLSPECIFIER __declspec(dllimport)
  #define EXPIMP_TEMPLATE extern
  #endif
#else
     #define DECLSPECIFIER
     #define EXPIMP_TEMPLATE
#endif