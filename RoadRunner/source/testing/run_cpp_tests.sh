
echo "Running tests..."

# ===  Call with three arguments!!
compiler=$1
compiler_version=$2
build_type=$3


install_folder=/home/sagrada/myhome/downloads/source/roadrunner/install
model_folder=$install_folder/all/models
wc=/home/sagrada/myhome/downloads/source/roadrunner/roadrunnerlib
report_file=$wc/reports/$compiler/$compiler_version/cpp_tests.xml
temp_folder=/tmp/$compiler_version

echo "cxx_api_tests -m$model_folder -r$report_file -lgcc -s$install_folder/ThirdParty/rr_support -t$temp_folder -d$temp_folder"

./cxx_api_tests -m$model_folder -r$report_file -lgcc -s$install_folder/ThirdParty/rr_support -t$temp_folder -d$temp_folder

echo "done..."
