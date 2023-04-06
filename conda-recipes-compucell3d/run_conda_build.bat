IF "%1"=="" (
    SET PYTHON_VERSION=3.7
) ELSE (
    SET PYTHON_VERSION=%1
)

conda build -c local -c conda-forge -c compucell3d . --python=%PYTHON_VERSION%
#conda render .