#if defined(_WIN32)
  #ifdef CAShared_EXPORTS
  #define CASHARED_EXPORT __declspec(dllexport)
  #define CASHARED_EXPIMP_TEMPLATE
  #else
    #define CASHARED_EXPORT __declspec(dllimport)
    #define CASHARED_EXPIMP_TEMPLATE extern
  #endif
#else
     #define CASHARED_EXPORT
     #define CASHARED_EXPIMP_TEMPLATE
#endif


