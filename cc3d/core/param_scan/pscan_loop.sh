# This script is provided as a very simple example on how to execute parameter scan using multiple workers
# (16 in out case)

# Important: make sure you change cc3d installation path and param scan output paths before using this script

# linux users: change paramScan.command to paramScan.sh

# Note: you can add command line options to time ${cc3d_install_dir}/paramScan.command  -i ...
# simply use the same commands you would use with paramScan script. USe parameters scan GUI (accessible from player)
# to generate for you command line options and then paste them here

export num_workers=16

export last_worker_id=`expr $num_workers - 1`

export cc3d_install_dir=/Users/m/Demo2/CC3D_4.1.1
export param_scan_output_dir=/Users/m/pscan_output

mkdir -p ${param_scan_output_dir}
for worker_id in $(seq 0 $last_worker_id)
do
    time ${cc3d_install_dir}/paramScan.command  -i ${cc3d_install_dir}/Demos/ParameterScan/CellSorting/CellSorting.cc3d -o ${param_scan_output_dir} -f 1000 --install-dir ${cc3d_install_dir} > ${param_scan_output_dir}/pscan_${worker_id}.out 2>&1 &
    echo "Worker ${worker_id}"
done
