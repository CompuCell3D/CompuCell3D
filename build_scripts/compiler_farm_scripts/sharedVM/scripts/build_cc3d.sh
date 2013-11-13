#!/bin/bash

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
        exit $status
    fi
    return $status    

}



time run_and_watch_status BUILDING_CC3D_370 ./build-cc3d-370-compiler-farm.sh
time run_and_watch_status BUILDING_RR_LLVM  ./build-rr-llvm-compiler-farm.sh 
time run_and_watch_status BUILDING_CC3D_371 ./build-cc3d-371-compiler-farm.sh
/sbin/poweroff


