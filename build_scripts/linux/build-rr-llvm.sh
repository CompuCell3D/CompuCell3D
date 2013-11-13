# example command ./build-debian-rr-llvm.sh -s=~/RR_LLVM_GIT -p=~/install_projects_RR_LLVM -c=4
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


export BUILD_ROOT=~/install_projects_RR_LLVM/build
export SOURCE_ROOT=~/RR_LLVM_GIT
export INSTALL_ROOT=~/install_projects_RR_LLVM
export DEPENDENCIES_ROOT=${INSTALL_ROOT}/depend
export INSTALL_PREFIX=${INSTALL_ROOT}/RR_LLVM


export BUILD_RR=NO
export BUILD_RR_DEPEND=NO
export BUILD_ALL=YES
export MAKE_MULTICORE=1


for i in "$@"
do
case $i in
    -p=*|--prefix=*)
    INSTALL_ROOT=`echo $i | sed 's/[-a-zA-Z0-9]*=//'`

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
    --rr)
    BUILD_ALL=NO
    BUILD_RR=YES
    ;;
    --rr-depend)
    BUILD_ALL=NO
    BUILD_RR_DEPEND=YES
    ;;
    --help)
    
    echo "build-debian-rr-llvm.sh <OPTIONS>"
    echo
    echo "OPTIONS:"
    echo
    echo "-p=<dir> | --install_root=<dir> : rr installation root (target directory) | default ~/install_projects/cc3d"
    echo
    echo "-b=<dir> | --build_root=<dir> : temporary directory for object files (aka compiler output) | default ~/install_projects/RR_LLVM_build"
    echo
    echo "-s=<dir> | --source_root=<dir> : root directory of RR_LLVM repository "
    echo
    echo "specifying options below will allow for selection of specific projects to build."
    echo  "If you are rebuilding CompuCell3D by picking specific projects you will shorten build time"
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
  BUILD_RR_DEPEND=YES
  BUILD_RR=YES
fi

echo BUILD_RR $BUILD_RR_DEPEND
echo BUILD_RR_DEPEND $BUILD_RR_DEPEND


# expanding paths
eval INSTALL_ROOT=$INSTALL_ROOT
eval INSTALL_PREFIX=$INSTALL_PREFIX
eval BUILD_ROOT=$BUILD_ROOT
eval SOURCE_ROOT=$SOURCE_ROOT
eval DEPENDENCIES_ROOT=$DEPENDENCIES_ROOT


BUILD_ROOT=${INSTALL_ROOT}/build
DEPENDENCIES_ROOT=${INSTALL_ROOT}/depend

echo INSTALL_ROOT = ${INSTALL_ROOT}
echo BUILD_ROOT = ${BUILD_ROOT}
echo SOURCE_ROOT = ${SOURCE_ROOT}
echo DEPENDENCIES_ROOT = ${DEPENDENCIES_ROOT}



echo MAKE_MULTICORE= $MAKE_MULTICORE
MAKE_MULTICORE_OPTION=-j$MAKE_MULTICORE
echo OPTION=$MAKE_MULTICORE_OPTION


mkdir -p $BUILD_ROOT
mkdir -p $DEPENDENCIES_ROOT


if [ "$BUILD_RR_DEPEND" == YES ]
then
  ############# BUILDING CC3D
  mkdir -p $BUILD_ROOT/third_party
  cd $BUILD_ROOT/third_party
  

  #checking if kernel is 32 or 64 bit
  # BITS=$( getconf LONG_BIT )

  # echo BITS = $BITS

  # OPTIMIZATION_FLAGS= 

  # if [ ! $BITS -eq 64 ]
  # then
      
      # # since you cannot pass cmake options that have spaces to cmake command line the alternatice is to prepare initial CmakeCache.txt file and put those options there...
      # echo 'CMAKE_C_FLAGS_RELEASE:STRING=-O0 -DNDEBUG'>>CMakeCache.txt
  # fi  
  

  run_and_watch_status THIRD_PARTY_CMAKE_CONFIG cmake -G "Unix Makefiles"  -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE:STRING=Release $SOURCE_ROOT/third_party
  run_and_watch_status THIRD_PARTY_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION VERBOSE=1 && make install
  ############# END OF BUILDING CC3D
fi

echo "THIS IS BUILD_RR $BUILD_RR"

if [ "$BUILD_RR" == YES ]
then
  ############# BUILDING CC3D
  mkdir -p $BUILD_ROOT/RR
  cd $BUILD_ROOT/RR	


  run_and_watch_status RR_CMAKE_CONFIG cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} -DBUILD_LLVM:BOOL=ON -DLLVM_CONFIG_EXECUTABLE:PATH=/usr/bin/llvm-config-3.2 -DBUILD_SWIG_PYTHON:BOOL=ON -DTHIRD_PARTY_INSTALL_FOLDER:PATH=${INSTALL_PREFIX}  -DCMAKE_BUILD_TYPE:STRING=Release $SOURCE_ROOT 
  run_and_watch_status RR_COMPILE_AND_INSTALL make $MAKE_MULTICORE_OPTION VERBOSE=1 && make install
  ############# END OF BUILDING CC3D
fi
