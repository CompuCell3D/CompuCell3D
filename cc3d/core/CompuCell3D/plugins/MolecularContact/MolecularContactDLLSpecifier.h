#ifndef MOLECULARCONTACT_EXPORT_H
#define MOLECULARCONTACT_EXPORT_H

    #if defined(_WIN32)
      #ifdef MolecularContactShared_EXPORTS
          #define MOLECULARCONTACT_EXPORT __declspec(dllexport)
          #define MOLECULARCONTACT_EXPIMP_TEMPLATE
      #else
          #define MOLECULARCONTACT_EXPORT __declspec(dllimport)
          #define MOLECULARCONTACT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define MOLECULARCONTACT_EXPORT
         #define MOLECULARCONTACT_EXPIMP_TEMPLATE
    #endif

#endif
