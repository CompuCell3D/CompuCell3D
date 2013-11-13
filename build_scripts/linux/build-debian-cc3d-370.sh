# example command ./build-debian-cc3d.sh -s=~/CC3D_GIT -p=~/install_projects/3.7.0 -c=4
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


export BUILD_ROOT=~/BuildCC3D
export SOURCE_ROOT=~/CODE_TGIT_NEW
export DEPENDENCIES_ROOT=~/install_projects
export INSTALL_PREFIX=~/install_projects/cc3d


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
    
    echo "build-debian-cc3d.sh <OPTIONS>"
    echo
    echo "OPTIONS:"
    echo
    echo "-p=<dir> | --prefix=<dir> : cc3d installation prefix (target directory) | default ~/install_projects/cc3d"
    echo
    echo "-b=<dir> | --build_root=<dir> : temporary directory for object files (aka compiler output) | default ~/BuildCC3D"
    echo
    echo "-s=<dir> | --source_root=<dir> : root directory of CC3D GIT repository "
    echo
    echo "-d=<dir> | --dependencies_root=<dir> : root directory where dependencies will be installed (used mainly  for bionet dependencies) | default ~/install_projects"
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

  run_and_watch_status COMPUCELL3D_CMAKE_CONFIG cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX $SOURCE_ROOT/CompuCell3D 
  run_and_watch_status COMPUCELL3D_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION && make install
  
  ############# END OF BUILDING CC3D
fi

if [ "$BUILD_BIONET_DEPEND" == YES ]
then
  ############# BUILDING SBML AND SUNDIALS BIONET DEPENDENCIES
  export CXXFLAGS=-fPIC
  export CFLAGS=-fPIC

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
  ############# END OF BUILDING SBML AND SUNDIALS BIONET DEPENDENCIES
fi

if [ "$BUILD_BIONET_DEPEND" == YES ]
then
  ############# BUILDING  BIONET 

  export BIONET_SOURCE=$SOURCE_ROOT/BionetSolver/0.0.6

  mkdir -p $BUILD_ROOT/BionetSolver
  cd $BUILD_ROOT/BionetSolver


  run_and_watch_status BIONET_CMAKE_CONFIG cmake -G "Unix Makefiles" -DLIBSBML_INSTALL_DIR:PATH=$DEPENDENCIES_ROOT/libsbml-3.4.1 -DSUNDIALS_INSTALL_DIR:PATH=$DEPENDENCIES_ROOT/sundials-2.3.0 -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX $BIONET_SOURCE
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
  

  #checking if kernel is 32 or 64 bit
  BITS=$( getconf LONG_BIT )

  echo BITS = $BITS

  OPTIMIZATION_FLAGS= 

  if [ ! $BITS -eq 64 ]
  then
      
      # since you cannot pass cmake options that have spaces to cmake command line the alternatice is to prepare initial CmakeCache.txt file and put those options there...
      echo 'CMAKE_C_FLAGS_RELEASE:STRING=-O0 -DNDEBUG'>>CMakeCache.txt
  fi  
  

  run_and_watch_status THIRD_PARTY_CMAKE_CONFIG cmake -G "Unix Makefiles"  -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}_RR -DCMAKE_BUILD_TYPE:STRING=Release $SOURCE_ROOT/RoadRunner/ThirdParty 
  run_and_watch_status THIRD_PARTY_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION VERBOSE=1 && make install
  ############# END OF BUILDING CC3D
fi

if [ "$BUILD_RR" == YES ]
then
  ############# BUILDING CC3D
  mkdir -p $BUILD_ROOT/RR
  cd $BUILD_ROOT/RR


  run_and_watch_status RR_CMAKE_CONFIG cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}_RR -DBUILD_CC3D_EXTENSION:BOOL=ON -DTHIRD_PARTY_INSTALL_FOLDER:PATH=${INSTALL_PREFIX}_RR -DCC3D_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE:STRING=Release $SOURCE_ROOT/RoadRunner 
  run_and_watch_status RR_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION VERBOSE=1 && make install
  ############# END OF BUILDING CC3D
fi
