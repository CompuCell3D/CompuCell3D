#!/bin/bash
#######################################################################################################
#                                   CC3D Linux installation script                                    #
#                                      Biocomplexity Institute                                        #
#                                      Indiana University                                             #
#                                      Bloomington, IN, U.S.A.                                        #
#                                                                                                     #
#                                      Written by T.J. Sego, Ph.D.                                    #
#######################################################################################################
#                                                                                                     #
# This script will download and install CC3D on your Linux machine                                    #
# You must specify the location of the CC3D installation below in the variable CC3D_ROOT_INSTALL_PATH #
# You can also select a version of CC3D to install                                                    #
#                                                                                                     #
#######################################################################################################
#                                            Configuration                                            #
#######################################################################################################
# Set this to where to install versions of CC3D
CC3D_ROOT_INSTALL_PATH=/cc3d
# Version info; only used if not building current state of repository
#   v4.2.0 through v4.2.2 are currently supported
CC3D_MAJOR_VERSION=4
CC3D_MINOR_VERSION=2
CC3D_BUILD_VERSION=2
# Optional build of current state of repository
#   Installation will be in CC3D_ROOT_INSTALL_PATH/vMaster
#   Installation will still reflect version info as specified above
#   Set to 1 to build from current master branch
BUILD_MASTER_BRANCH=0
#######################################################################################################



# Root installation directory
if [ "$BUILD_MASTER_BRANCH" -eq 1 ] ; then
  # Master branch
  PREFIX_VERSION=vMaster
else
  # Version
  PREFIX_VERSION=${CC3D_MAJOR_VERSION}${CC3D_MINOR_VERSION}${CC3D_BUILD_VERSION}
fi
CC3D_INSTALL_PATH=${CC3D_ROOT_INSTALL_PATH}/${PREFIX_VERSION}
# CC3D Version
CC3D_VERSION=${CC3D_MAJOR_VERSION}${CC3D_MINOR_VERSION}${CC3D_BUILD_VERSION}
CC3D_VERSION_DOTTED=${CC3D_MAJOR_VERSION}.${CC3D_MINOR_VERSION}.${CC3D_BUILD_VERSION}
# Repo clone directory
SOURCE_DIR=${CC3D_INSTALL_PATH}/CC3D_PY3_GIT
# Installation directory
SUB_INSTALL_PATH=${CC3D_INSTALL_PATH}/install
# Build directory
SUB_BUILD_PATH=${CC3D_INSTALL_PATH}/build
# Build scripts directory
SUB_BUILD_SCRIPTS_PATH=${CC3D_INSTALL_PATH}/build_scripts

# Python version
PYTHON_VERSION=3.7
# Repository source
REPO_URL=https://github.com/CompuCell3D/CompuCell3D.git

# Color coding
FG_COLOR=18
FG_SEQ="38;5;${FG_COLOR}"
BG_COLOR=28
BG_SEQ="48;5;${BG_COLOR}"
CC3D_TXT="\e[1;${FG_SEQ};${BG_SEQ}mCompuCell3D\e[0m"

ERROR_TXT="\e[1;91mError\e[0m"


if [ "$BUILD_MASTER_BRANCH" -eq 1 ] ; then
  DISP_VERSION="master version"
else
  DISP_VERSION=v${CC3D_VERSION_DOTTED}
fi

echo ""
echo "################################################"
echo -e " ${CC3D_TXT} installation (${DISP_VERSION})"
echo " Brought to you by the Biocomplexity Institute"
echo " at Indiana University"
echo "################################################"
echo ""

# Prep

#   Remove old directories
if [ -d "$CC3D_INSTALL_PATH" ] ; then

  echo -e "${ERROR_TXT}: previous installation of ${CC3D_TXT} ${DISP_VERSION} detected at ${CC3D_INSTALL_PATH}"
  echo "First remove this directory and then try again"
  exit 1

fi

#   Make fresh directories
mkdir "$CC3D_INSTALL_PATH"
mkdir "$SOURCE_DIR"
mkdir "$SUB_BUILD_SCRIPTS_PATH"

# Create conda environment
if [ "$BUILD_MASTER_BRANCH" -eq 1 ] ; then
  #   Master branch
  CONDA_NAME=cc3d_master
else
  #   Version
  CONDA_NAME=cc3d_${CC3D_VERSION}
fi

#   Get info about environment
#   If it exists and we're NOT building master, then leave it alone
#   If it exists and we ARE building master, then remove it
source activate ${CONDA_NAME}> /dev/null 2>&1
if [ $? -eq 0 ] ; then

  source deactivate ${CONDA_NAME}

  if [ "$BUILD_MASTER_BRANCH" -eq 1 ] ; then
    echo "Removing previous installation of conda environment ${CONDA_NAME}..."
    printf 'y\n' | conda env remove -n ${CONDA_NAME}> /dev/null
    creating_env=1
  else
    creating_env=0
  fi

else
  creating_env=1
fi

#   Create environment if necessary
if [ ${creating_env} -eq 1 ] ; then
  echo "Intalling conda environment ${CONDA_NAME}..."

  printf 'y\n' | conda create -n ${CONDA_NAME} python=${PYTHON_VERSION}
  if [ $? -ne 0 ] ; then
    echo -e "${ERROR_TXT}: Something went wrong during ${CC3D_TXT} conda environment creation"
    exit 4
  fi

fi

# Clone build scripts
echo "Cloning build scripts..."

cd ${SUB_BUILD_SCRIPTS_PATH} || exit 99

git clone https://github.com/CompuCell3D/cc3d_build_scripts.git
if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during clone of ${CC3D_TXT} build scripts"
  exit 2
fi

# Clone source from repo
echo "Cloning repo..."

cd ${SOURCE_DIR} || exit 99

if [ "$BUILD_MASTER_BRANCH" -eq 1 ] ; then
  #   Master branch
  git clone "$REPO_URL" || git pull
else
  #   Version
  BRANCH_NAME=release/${CC3D_VERSION_DOTTED}
  echo "  Cloning branch ${BRANCH_NAME}"
  git clone --branch ${BRANCH_NAME} "$REPO_URL" || git pull
fi

if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during clone of ${CC3D_TXT} source code"
  exit 3
fi

# Install dependencies
echo -e "Installing ${CC3D_TXT} dependencies..."

source activate ${CONDA_NAME}> /dev/null

printf 'y\n' | conda install -c conda-forge numpy scipy pandas jinja2 webcolors vtk=8.2 pyqt pyqtgraph deprecated qscintilla2 jinja2 chardet cmake swig=3 requests
if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during installation of ${CC3D_TXT} dependencies"
  exit 5
fi

printf 'y\n' | conda install -c compucell3d tbb_full_dev
if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during installation of TBB"
  exit 6
fi

pip install libroadrunner
if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during installation of libroadrunner"
  exit 7
fi

pip install antimony
if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during installation of Antimony"
  exit 8
fi

# Compile
echo -e "Compling ${CC3D_TXT}..."

source activate root
cd "${SUB_BUILD_SCRIPTS_PATH}/cc3d_build_scripts/linux/400" || exit 99
python build.py --prefix=${SUB_INSTALL_PATH} --source-root=${SOURCE_DIR}/CompuCell3D --build-dir=${SUB_BUILD_PATH} --version=${CC3D_VERSION_DOTTED} --conda-env-name=${CONDA_NAME}
if [ $? -ne 0 ] ; then
  echo -e "${ERROR_TXT}: Something went wrong during compilation!"
  exit 9
fi

# Report
echo -e "${CC3D_TXT} successfully installed!"
echo "Installation directory: ${CC3D_INSTALL_PATH}"
exit 0
