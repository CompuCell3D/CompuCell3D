cmake_minimum_required(VERSION 2.8)

#These will be the defaults ...
if(WIN32)
SET(RR_BUILDS_ROOT      "c:/builds")
SET(RR_INSTALLS_ROOT    "c:/installs")
else()
SET(RR_BUILDS_ROOT      "~/builds")
SET(RR_INSTALLS_ROOT    "~/installs")
endif()


if(${BORLAND})
set(FOLDER_POSTFIX "xe")
else(${BORLAND})
set(FOLDER_POSTFIX "vs")
endif()


set(THIRD_PARTY_INSTALL_FOLDER ${RR_INSTALLS_ROOT}/${FOLDER_POSTFIX}/release CACHE PATH "Directory containing ThirdParty libraries.")
if("${THIRD_PARTY_INSTALL_FOLDER}" STREQUAL "")
  set(THIRD_PARTY_INSTALL_FOLDER ${RR_INSTALLS_ROOT}/${FOLDER_POSTFIX}/release CACHE PATH "Directory containing ThirdParty libraries" FORCE)
endif()

add_subdirectory(Poco)
