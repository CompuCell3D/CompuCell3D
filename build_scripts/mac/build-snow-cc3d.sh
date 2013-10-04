# example command ./build-osx-cc3d.sh -s=~/CC3D_GIT -p=~/install_projects/CC3D_3.7.0 -r=~/RR_OSX -d=/Users/Shared/CC3Ddev/Dependencies -b=CC3D_3.7.0_MacOSX_10.8_32bit -c=4
#command line parsing

function run_and_watch_status {
    # first argument is a task descriptor and is mandatory
    # the remaining arguments make up a command to execute
    
    #executing the command
    "${@:2}"
    # querrying its status
    status=$?
    echo "STATUS=$status"
    if [ $status -ne 0 ]; then
        echo "error with $1"
        exit
    fi
    return $status    

}


export BUILD_ROOT=
export SOURCE_ROOT=~/CC3D_GIT
export DEPENDENCIES_ROOT=
export INSTALL_PREFIX=~/install_projects/cc3d
export RR_SOURCE_ROOT=~/RR_OSX
#mac variables
export GCC_DIR=~/Deps/gcc_4.6.0
export VTK_BIN_AND_BUILD_DIR=~/Deps/VTK_5.8.0_bin_and_build
export MAC_DEPS=~/Deps/OSX_Leopard_Deps                 
export OUTPUT_BINARY_NAME=CC3D_3.7.0_MacOSX_10.6_32bit

export BUILD_CC3D=NO
export BUILD_BIONET=NO
export BUILD_BIONET_DEPEND=NO
export BUILD_CELLDRAW=NO
export BUILD_RR=NO
export BUILD_RR_DEPEND=NO
export BUILD_ALL=YES
export MAKE_MULTICORE=1


for i in "$@"
do
case $i in
    -p=*|--prefix=*)
    INSTALL_PREFIX=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`

    ;;
    -s=*|--source-root=*)
    SOURCE_ROOT=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`
    ;;

    -r=*|--rr-source-root=*)
    RR_SOURCE_ROOT=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`
    ;;

    -d=*|--mac-dependencies=*)
    MAC_DEPS=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`
    ;;
    -b=*|--output-binary-name=*)
    OUTPUT_BINARY_NAME=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`
    ;;

    
    -c=*|--cores=*)
    MAKE_MULTICORE=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`
    ;;
    --cc3d)
    BUILD_ALL=NO
    BUILD_CC3D=YES
    ;;
    --bionet)
    BUILD_ALL=NO
    BUILD_BIONET=YES
    ;;
    --bionet-depend)
    BUILD_ALL=NO
    BUILD_BIONET_DEPEND=YES
    ;;
    --celldraw)
    BUILD_ALL=NO
    BUILD_CELLDRAW=YES
    ;;
    --rr)
    BUILD_ALL=NO
    BUILD_RR=YES
    ;;
    --rr-depend)
    BUILD_ALL=NO
    BUILD_RR_DEPEND=YES
    ;;
    --help)
    
    echo "build-osx-cc3d.sh <OPTIONS>"
    echo
    echo "OPTIONS:"
    echo
    echo "-p=<dir> | --prefix=<dir> : cc3d installation prefix (target directory) | default ~/install_projects/cc3d"
    echo
    echo "-r=<dir> | --rr-source-root=<dir> : RoadRunner Source"
    echo
    echo "-s=<dir> | --source-root=<dir> : root directory of CC3D GIT repository "
    echo
    echo "specifying options below will allow for selection of specific projects to build."
    echo  "If you are rebuilding CompuCell3D by picking specific projects you will shorten build time"
    echo
    echo "--cc3d : builds CompuCell3D"
    echo
    echo "--bionet : builds BionetSolver"
    echo
    echo "--bionet-depend : builds BionetSolver dependencies"
    echo
    echo "--celldraw : builds celldraw"
    echo
    echo "--rr : builds RoadRunner"
    echo
    echo "--rr-depend : builds RoadRunner dependencies"
    echo
    ;;
    
    *)
            # unknown option
    ;;
esac
done



if [ "$BUILD_ALL" == YES ]
then
  BUILD_CC3D=YES
  BUILD_BIONET=YES
  BUILD_BIONET_DEPEND=YES
  BUILD_CELLDRAW=YES
  BUILD_RR=YES
  BUILD_RR_DEPEND=YES
  BUILD_ALL=YES
fi

echo BUILD_CC3D $BUILD_CC3D
echo BUILD_BIONET $BUILD_BIONET
echo BUILD_BIONET_DEPEND $BUILD_BIONET_DEPEND
echo BUILD_CELLDRAW $BUILD_CELLDRAW
echo BUILD_RR $BUILD_RR_DEPEND
echo BUILD_RR_DEPEND $BUILD_RR_DEPEND


# expanding paths
eval INSTALL_PREFIX=$INSTALL_PREFIX
eval BUILD_ROOT=$BUILD_ROOT
eval SOURCE_ROOT=$SOURCE_ROOT
eval DEPENDENCIES_ROOT=$DEPENDENCIES_ROOT


eval GCC_DIR=${GCC_DIR}
eval VTK_BIN_AND_BUILD_DIR=${VTK_BIN_AND_BUILD_DIR}
eval MAC_DEPS=${MAC_DEPS}                 




BUILD_ROOT=${INSTALL_PREFIX}_build
DEPENDENCIES_ROOT=${INSTALL_PREFIX}_depend

echo INSTALL_PREFIX = ${INSTALL_PREFIX}
echo BUILD_ROOT = ${BUILD_ROOT}
echo SOURCE_ROOT = ${SOURCE_ROOT}
echo DEPENDENCIES_ROOT = ${DEPENDENCIES_ROOT}



echo MAKE_MULTICORE= $MAKE_MULTICORE
MAKE_MULTICORE_OPTION=-j$MAKE_MULTICORE
echo OPTION=$MAKE_MULTICORE_OPTION


mkdir -p $BUILD_ROOT
mkdir -p $DEPENDENCIES_ROOT


if [ "$BUILD_CC3D" == YES ]
then
  ############# BUILDING CC3D
  mkdir -p $BUILD_ROOT/CompuCell3D
  cd $BUILD_ROOT/CompuCell3D


 

  run_and_watch_status COMPUCELL3D_CMAKE_CONFIG cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX -DNO_OPENCL:BOOL=ON -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.5 -DPYTHON_MINOR_VERSION:STRING=5 -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python2.5 -DPYTHON_INCLUDE_DIR:PATH=/System/Library/Frameworks/Python.framework/Versions/2.5/Headers -DPYTHON_LIBRARY:FILEPATH=/usr/lib/libpython2.5.dylib -DEIGEN3_INCLUDE_DIR=${SOURCE_ROOT}/CompuCell3D/core/Eigen -DCMAKE_C_COMPILER:FILEPATH=${GCC_DIR}/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=${GCC_DIR}/bin/g++ PATH=$INSTALL_PREFIX  -DVTK_DIR:PATH=${VTK_BIN_AND_BUILD_DIR}/lib/vtk-5.8 -DCMAKE_CXX_FLAGS="-mmacosx-version-min=10.5 -O3 -g -m32" -DCMAKE_C_FLAGS="-mmacosx-version-min=10.5 -O3 -g -m32" $SOURCE_ROOT/CompuCell3D
  run_and_watch_status COMPUCELL3D_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION  VERBOSE=1 && make install
  
  ############# END OF BUILDING CC3D
fi


################### INSTALLING DEPENDENCIES

cp -a ${MAC_DEPS}/* ${INSTALL_PREFIX}


################## END OF INSTALLING DEPENDENCIES


if [ "$BUILD_BIONET_DEPEND" == YES ]
then
  ############# BUILDING SBML AND SUNDIALS BIONET DEPENDENCIES

  # bionet solver deps are built using standard OSX gcc compilers
  export CXXFLAGS="-fPIC -arch x86_64 -arch i386"
  export CFLAGS="-fPIC -arch x86_64 -arch i386"
  export LDFLAGS="-arch x86_64 -arch i386" 


  SBML_BUILD_DIR=$BUILD_ROOT/libsbml-3.4.1
  SBML_INSTALL_DIR=$DEPENDENCIES_ROOT/libsbml-3.4.1

  
  if [ ! -d "$SBML_INSTALL_DIR" ]; then # SBML_INSTALL_DIR does not exist
  
    cp $SOURCE_ROOT/BionetSolver/dependencies/libsbml-3.4.1-src.zip $BUILD_ROOT
    cd $BUILD_ROOT 

    unzip libsbml-3.4.1-src.zip
    
    cd $SBML_BUILD_DIR 
    run_and_watch_status LIBSBML_CONFIGURE ./configure --prefix=$SBML_INSTALL_DIR 
    # libsbml does not compile well with multi-core option on
    run_and_watch_status LIBSBML_COMPILE_AND_INSTALL make  && make install
  fi


  SUNDIALS_BUILD_DIR=$BUILD_ROOT/sundials-2.3.0
  SUNDIALS_INSTALL_DIR=$DEPENDENCIES_ROOT/sundials-2.3.0
  
  if [ ! -d "$SUNDIALS_INSTALL_DIR" ]; then # SUNDIALS_INSTALL_DIR does not exist
    
    cp $SOURCE_ROOT/BionetSolver/dependencies/sundials-2.3.0.tar.gz $BUILD_ROOT
    cd $BUILD_ROOT

    tar -zxvf sundials-2.3.0.tar.gz

    cd $SUNDIALS_BUILD_DIR
    run_and_watch_status SUNDIALS_CONFIGURE ./configure --with-pic --prefix=$SUNDIALS_INSTALL_DIR

    run_and_watch_status SUNDIALS_COMPILE_AND_INSTALL  make $MAKE_MULTICORE_OPTION && make install
  fi

  # reset flags
  CXXFLAGS=
  CFLAGS= 
  LDFLAGS=

  ############# END OF BUILDING SBML AND SUNDIALS BIONET DEPENDENCIES
fi



if [ "$BUILD_BIONET_DEPEND" == YES ]
then
  ############# BUILDING  BIONET 

  export BIONET_SOURCE=$SOURCE_ROOT/BionetSolver/0.0.6

  mkdir -p $BUILD_ROOT/BionetSolver
  cd $BUILD_ROOT/BionetSolver


  run_and_watch_status BIONET_CMAKE_CONFIG cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.5 -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python2.5 -DPYTHON_INCLUDE_DIR:PATH=/System/Library/Frameworks/Python.framework/Versions/2.5/Headers -DPYTHON_LIBRARY:FILEPATH=/usr/lib/libpython2.5.dylib -DCMAKE_C_COMPILER:FILEPATH=${GCC_DIR}/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=${GCC_DIR}/bin/g++ PATH=$INSTALL_PREFIX   -DCMAKE_CXX_FLAGS="-mmacosx-version-min=10.5 -O3 -g -m32" -DCMAKE_C_FLAGS="-mmacosx-version-min=10.5 -O3 -g -m32" -DLIBSBML_INSTALL_DIR:PATH=$DEPENDENCIES_ROOT/libsbml-3.4.1 -DSUNDIALS_INSTALL_DIR:PATH=$DEPENDENCIES_ROOT/sundials-2.3.0 DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX $BIONET_SOURCE
  # -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.6 -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python2.6 -DPYTHON_INCLUDE_DIR:PATH=/System/Library/Frameworks/Python.framework/Versions/2.6/Headers -DEIGEN3_INCLUDE_DIR=${SOURCE_ROOT}/CompuCell3D/core/Eigen -DPYTHON_LIBRARY:FILEPATH=/usr/lib/libpython2.6.dylib -DCMAKE_C_COMPILER:FILEPATH=${GCC_DIR}/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=${GCC_DIR}/bin/g++ PATH=$INSTALL_PREFIX  -DVTK_DIR:PATH=${VTK_BIN_AND_BUILD_DIR}/lib/vtk-5.8 -DCMAKE_CXX_FLAGS="-mmacosx-version-min=10.6 -O3 -g -fpermissive -m64" -DCMAKE_C_FLAGS="-mmacosx-version-min=10.6 -O3 -g -fpermissive -m64"
  run_and_watch_status BIONET_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION && make install

  ############# END OF BUILDING  BIONET 
fi

if [ "$BUILD_CELLDRAW" == YES ]
then
  ############# BUILDING  CELLDRAW 
  export CELLDRAW_SOURCE=$SOURCE_ROOT/CellDraw/1.5.1

  mkdir -p $BUILD_ROOT/CellDraw
  cd $BUILD_ROOT/CellDraw


  run_and_watch_status CELLDRAW_CMAKE_CONFIG cmake -G "Unix Makefiles"  -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX $CELLDRAW_SOURCE
  run_and_watch_status CELLDRAW_COMPILE_AND_INSTALL make && make install
  ############# END OF  CELLDRAW 
fi



if [ "$BUILD_RR_DEPEND" == YES ]
then
  ############# BUILDING CC3D
  mkdir -p $BUILD_ROOT/RRDepend
  cd $BUILD_ROOT/RRDepend
  export CC=${GCC_DIR}/bin/gcc
  export CXX=${GCC_DIR}/bin/g++


  run_and_watch_status THIRD_PARTY_CMAKE_CONFIG cmake -G "Unix Makefiles"  -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}_RR -DCMAKE_BUILD_TYPE:STRING=Release ${RR_SOURCE_ROOT}/third_party 
  # run_and_watch_status THIRD_PARTY_CMAKE_CONFIG cmake -G "Unix Makefiles"  -DCMAKE_OSX_ARCHITECTURES="x86_64" -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}_RR -DCMAKE_BUILD_TYPE:STRING=Release ${RR_SOURCE_ROOT}/third_party 
  run_and_watch_status THIRD_PARTY_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION VERBOSE=1 && make install
  

  CC=
  CXX=

  ############# END OF BUILDING CC3D
fi

if [ "$BUILD_RR" == YES ]
then
  ############# BUILDING CC3D
  mkdir -p $BUILD_ROOT/RR
  cd $BUILD_ROOT/RR

  export CC=${GCC_DIR}/bin/gcc
  export CXX=${GCC_DIR}/bin/g++

  run_and_watch_status RR_CMAKE_CONFIG cmake -G "Unix Makefiles" -DPYTHON_EXECUTABLE:PATH="/usr/bin/python2.5" -DPYTHON_INCLUDE_DIR:PATH="/System/Library/Frameworks/Python.framework/Versions/2.5/Headers" -DPYTHON_LIBRARY="/usr/lib/libpython2.5.dylib" -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}_RR -DBUILD_CC3D_EXTENSION:BOOL=ON -DTHIRD_PARTY_INSTALL_FOLDER:PATH=${INSTALL_PREFIX}_RR -DCC3D_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE:STRING=Release ${RR_SOURCE_ROOT} 
  # run_and_watch_status RR_CMAKE_CONFIG cmake -G "Unix Makefiles" -DCMAKE_OSX_ARCHITECTURES="x86_64" -DPYTHON_EXECUTABLE:PATH="/usr/bin/python2.6" -DPYTHON_INCLUDE_DIR:PATH="/System/Library/Frameworks/Python.framework/Versions/2.6/Headers" -DPYTHON_LIBRARY="/usr/lib/libpython2.6.dylib" -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}_RR -DBUILD_CC3D_EXTENSION:BOOL=ON -DTHIRD_PARTY_INSTALL_FOLDER:PATH=${INSTALL_PREFIX}_RR -DCC3D_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE:STRING=Release ${RR_SOURCE_ROOT} 
  run_and_watch_status RR_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION VERBOSE=1 && make install

  CC=
  CXX=

  ############# END OF BUILDING CC3D
fi

################### BUILDING ZIP-BASED INSTALLER



cd $INSTALL_PREFIX
cd ..
# date


DATE_FORMAT=$(date +"%Y%m%d")
echo $DATE_FORMAT
CC3D_ARCHIVE=${OUTPUT_BINARY_NAME}_${DATE_FORMAT}.zip
echo CC3D_ARCHIVE $CC3D_ARCHIVE

rm -f ${CC3D_ARCHIVE}
run_and_watch_status ZIPPING_BINARY ditto -c -k --keepParent -rsrcFork $INSTALL_PREFIX ${CC3D_ARCHIVE}

################### END OF BUILDING ZIP-BASED INSTALLER



# DATE_FORMAT= eval date +"%Y%m%d"
# echo THIS IS DATE FORMAT ${DATE_FORMAT}

# # echo ${OUTPUT_BINARY_NAME}_${DATE_FORMAT}.zip

# # CC3D_ARCHIVE="${OUTPUT_BINARY_NAME}_${DATE_FORMAT}.zip"
# CC3D_ARCHIVE="${DATE_FORMAT}"
# echo CC3D_ARCHIVE_NAME ${CC3D_ARCHIVE}
# rm -f ${CC3D_ARCHIVE}
# # run_and_watch_status ZIPPING_BINARY  zip -r ${OUTPUT_BINARY_NAME}_${DATE_FORMAT}.zip $INSTALL_PREFIX
