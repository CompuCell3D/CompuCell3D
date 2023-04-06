#if defined(_WIN32)
  #ifdef LoggerShared_EXPORTS
  #define LOGGER_EXPORT __declspec(dllexport)
  #define LOGGER_EXPIMP_TEMPLATE
  #else
  #define LOGGER_EXPORT __declspec(dllimport)
  #define LOGGER_EXPIMP_TEMPLATE extern
  #endif
#else
     #define LOGGER_EXPORT
     #define LOGGER_EXPIMP_TEMPLATE
#endif