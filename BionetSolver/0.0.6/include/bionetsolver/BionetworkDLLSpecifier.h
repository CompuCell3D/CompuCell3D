#ifndef BIONETWORK_EXPORT_H
#define BIONETWORK_EXPORT_H

    #if defined(_WIN32)
      #ifdef SBML_ODESOLVER_EXPORTS
          #define BIONETWORK_EXPORT __declspec(dllexport)
          #define BIONETWORK_EXPIMP_TEMPLATE
      #else
          #define BIONETWORK_EXPORT __declspec(dllimport)
          #define BIONETWORK_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define BIONETWORK_EXPORT
         #define BIONETWORK_EXPIMP_TEMPLATE
    #endif

#endif
