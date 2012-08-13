#ifndef XMLPARSEREXPATLIB_EXPORT_H
#define XMLPARSEREXPATLIB_EXPORT_H

    #if defined(_WIN32)
      #ifdef XMLParserExpatLibShared_EXPORTS
          #define XMLPARSEREXPATLIB_EXPORT __declspec(dllexport)
          #define XMLPARSEREXPATLIB_EXPIMP_TEMPLATE
      #else
          #define XMLPARSEREXPATLIB_EXPORT __declspec(dllimport)
          #define XMLPARSEREXPATLIB_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define XMLPARSEREXPATLIB_EXPORT
         #define XMLPARSEREXPATLIB_EXPIMP_TEMPLATE
    #endif

#endif
