if [ $# -eq 0 ]; then
    # If not, assign a default value
    PYTHON_VERSION=3.12
else
    # If yes, use the passed value
    PYTHON_VERSION="$1"
fi

check_boa() {
  if command -v boa >/dev/null 2>&1; then
    return 1
  else
    return 0
  fi
}

# Call the function
check_boa

# Store the exit code in a variable
exit_code=$?

if [ $exit_code -eq 1 ]; then
  echo "boa is available in PATH"
  conda mambabuild -c local -c conda-forge -c compucell3d . --python="$PYTHON_VERSION"
else
  echo "boa is not available in PATH"
  conda mambabuild -c local -c conda-forge -c compucell3d . --python="$PYTHON_VERSION"

fi
# conda render .