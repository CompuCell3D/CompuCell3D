#!/bin/bash

# function WaitForExit {

# 	local wait_for_exit=1
# 	echo $wait_for_exit
# 	local COUNTER=0
# 	# while [  $wait_for_exit ]; do
# 	# while [  $COUNTER -le 10 ]; do	
# 	while [  $wait_for_exit -eq 1 ]; do

# 		machine_running=$(VBoxManage list runningvms |grep ${1}|wc -l)

# 		echo "machine_running="${machine_running}
# 		if [ $machine_running -ge 1 ]; then
# 			echo "MACHINE RUNNING"
# 		else
# 			echo "MACHINE POWERED OFF"			
# 		fi 

# 		let COUNTER+=1
# 		# echo " compare ", { $COUNTER -eq 10 }
# 		if [ $COUNTER -eq 10 ]; then
			
# 			wait_for_exit=0
# 			echo "wait_for_exit=" $wait_for_exit
# 			# break
# 		fi
# 		echo "WAITING FOR EXIT ", $COUNTER, " system=",$1 " wait_for_exit="$wait_for_exit
# 	done
# 	echo "FINAL wait_for_exit=" $wait_for_exit	
# }

function WaitForExit {

	local wait_for_exit=1
	echo $wait_for_exit
	local COUNTER=0
	# while [  $wait_for_exit ]; do
	# while [  $COUNTER -le 10 ]; do	
	sleep 5
	while [  $wait_for_exit -eq 1 ]; do

		machine_running=$(VBoxManage list runningvms |grep ${1}|wc -l)

		echo "machine_running="${machine_running}
		if [ $machine_running -ge 1 ]; then
			echo "MACHINE ${1} RUNNING"
			sleep 60
		else
			wait_for_exit=0
			echo "MACHINE ${1} POWERED OFF"			
		fi 

	done
	# echo "FINAL wait_for_exit=" $wait_for_exit	
}


function start_compilation {
	local machine=$1
	echo "STARTING VIRTUAL MACHINE "$machine
	VBoxManage startvm "${machine}"
}



# start_compilation kubuntu_13_04



vb_machines=(kubuntu_13_10 kubuntu_13_10_64 kubuntu_13_04 kubuntu_13_04_64 kubuntu_12_10 kubuntu_12_10_64 kubuntu_12_04 kubuntu_12_04_64 kubuntu_11_10 kubuntu_11_10_64)
# vb_machines=(kubuntu_12_10_64)

for machine in ${vb_machines[@]}
do
	echo "THIS IS MY MACHINE="$machine
	start_compilation $machine
	WaitForExit $machine
done 


#WaitForExit kubuntu_13_04
