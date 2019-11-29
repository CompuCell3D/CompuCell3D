export num_workers=16
export last_worker_id=`expr $num_workers - 1`
echo $last_worker
for worker_id in $(seq 0 $last_worker_id)
do
    time /Users/m/Demo2/CC3D_4.1.0/paramScan.command  -i /Users/m/Demo2/CC3D_4.1.0/Demos/ParameterScan/CellSorting/CellSorting.cc3d -o /Users/m/Demo2/CC3D_4.1.0/pscan -f 1000 --install-dir /Users/m/Demo2/CC3D_4.1.0 > /Users/m/Demo2/CC3D_4.1.0/pscan_${worker_id}.out 2>&1 &
    echo "Worker ${worker_id}"
done

#time /Users/m/Demo2/CC3D_4.1.0/paramScan.command  -i /Users/m/Demo2/CC3D_4.1.0/Demos/ParameterScan/CellSorting/CellSorting.cc3d -o /Users/m/Demo2/CC3D_4.1.0/pscan -f 1000 --install-dir /Users/m/Demo2/CC3D_4.1.0 > /Users/m/Demo2/CC3D_4.1.0/pscan_0.out 2>&1 &
#time /Users/m/Demo2/CC3D_4.1.0/paramScan.command  -i /Users/m/Demo2/CC3D_4.1.0/Demos/ParameterScan/CellSorting/CellSorting.cc3d -o /Users/m/Demo2/CC3D_4.1.0/pscan -f 1000 --install-dir /Users/m/Demo2/CC3D_4.1.0 > /Users/m/Demo2/CC3D_4.1.0/pscan_1.out 2>&1 &