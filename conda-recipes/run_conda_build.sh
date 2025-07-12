if [ $# -eq 0 ]; then
    # If not, assign a default value
    PYTHON_VERSION=3.12
else
    # If yes, use the passed value
    PYTHON_VERSION="$1"
fi

check_boa() {
  if command -v boa >/dev/null 2>&1; then
    echo 1
  else
    echo 0
  fi
}

# Call the function
check_boa
# Store the exit code in a variable
exit_code=$(check_boa)
echo "BOA = $exit_code"
#exit_code=$?


# Function to extract the Conda platform architecture
conda_architecture() {
    # Run conda info command and store the output in a variable
    local output=$(conda info)

    # Extract the platform value using awk
    local platform=$(echo "$output" | awk -F': ' '/platform/ {print $2}')

    # Trim any leading or trailing spaces from the platform value
    platform=$(echo "$platform" | awk '{$1=$1};1')

    # Print the platform value
    echo "$platform"
}

architecture=$(conda_architecture)
echo "Conda Architecture: $architecture"



config_yaml=conda_build_config_x86.yaml

if [[ "$architecture" == "osx-arm64" ]]; then
    config_yaml=conda_build_config_arm64.yaml
fi

echo $config_yaml

if [ $exit_code -eq 1 ]; then
  echo "boa is available in PATH"
#  conda render . -e "$config_yaml"
    conda mambabuild -c conda-forge -c compucell3d . --python="$PYTHON_VERSION" -e "$config_yaml"
else
  echo "boa is not available in PATH"
#  conda render . -e "$config_yaml"
  conda mambabuild -c conda-forge -c compucell3d . --python="$PYTHON_VERSION" -e "$config_yaml"

fi


#conda mambabuild -c conda-forge -c compucell3d . --python="$PYTHON_VERSION"
##conda render .