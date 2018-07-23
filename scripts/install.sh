
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

PROJECT_DIR="${SCRIPT_DIR}/.."

source "${SCRIPT_DIR}/dbash.sh" || exit 1

cd ${SCRIPT_DIR}

PYENV="${SCRIPT_DIR}/../py_env"
if [[ ! -e ${PYENV} ]];then
    dbash::pp "# We setup a virtual environment for this project!"
    if ! dbash::command_exists virtualenv;then
        dbash::pp "# We install virtualenv!"
        sudo pip install virtualenv
    fi
    virtualenv -p python3 ${PYENV} --clear
    virtualenv -p python3 ${PYENV} --relocatable
fi

source ${PYENV}/bin/activate

dbash::pp "# Should we upgrade all dependencies?"
dbash::user_confirm ">> Update dependencies?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${PYENV}/bin/pip install --upgrade pip
    ${PYENV}/bin/pip install --upgrade numpy scipy matplotlib joblib ipdb python-gflags google-apputils autopep8 sklearn
    ${PYENV}/bin/pip install --upgrade pandas ipython ipdb jupyter opencv-python h5py keras
fi

dbash::user_confirm ">> Install tensorflow (CPU-only)?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${PYENV}/bin/pip install tensorflow
fi

dbash::user_confirm ">> Install tensorflow (GPU)?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    dbash::pp "Please install cuda 8.0 from nvidia!"
    dbash::pp "Please install cudnn 5.0 from nvidia!"
    dbash::pp "Notice, symbolic links for libcudnn.dylib and libcuda.dylib have to be added."
    ${PYENV}/bin/pip install tensorflow-gpu
fi

python setup.py install
python -m ipykernel install --user --name=py_env
