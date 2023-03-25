if [ $# -eq 0 ]; then
    # If not, assign a default value
    PYTHON_VERSION=3.7
else
    # If yes, use the passed value
    PYTHON_VERSION="$1"
fi


conda build -c local -c conda-forge -c compucell3d . --python="$PYTHON_VERSION"
#conda render .