#!/bin/sh

arguments=("$@")

cpu=false
gpu=false
for arg in "${arguments[@]}"; do
    if   [ "$arg" == "cpu" ];  then
        cpu=true
    elif [ "$arg" == "gpu" ];  then
        gpu=true
    else
        warning "Argument '$arg' was not understood."
	return 1
    fi
done

# Check whether conda is installed
if ! hash conda 2>/dev/null; then
    print "conda was not installed."
    question "Do you want to do it now?"
    response="$?"
    if (( "$response" )); then

	# Install miniconda
	echo setting Conda package from /home/codes/Python-Anaconda/py36/etc/profile.d/conda.sh
        source /home/codes/Python-Anaconda/py36/etc/profile.d/conda.sh

	# Check if installation succeeded.
	response="$?"
	if (( "$response" )); then
	    return 1
	fi
	
    else
	warning "Please install conda manually, see e.g. https://gitlab.epfl.ch/spc/tcv/event-detection/-/tree/UTime-PlasmaStates-V2. Exiting."
	return 1
    fi
fi

if $gpu; then
    env="confinement-gpu"
elif $cpu; then
    env="confinement-cpu"
fi

echo creating an empty Conda environment
mkdir -p $HOME/NoTivoli/$env
conda create --prefix $HOME/NoTivoli/$env

echo activating Conda environment
conda activate $HOME/NoTivoli/$env

echo installing packages
conda install -c anaconda python=3.7
conda install -c conda-forge ruamel.yaml

echo installation of UTime packages
pip install -e algorithms/GMUTime/UTime-PlasmaStates
pip install h5py==2.10.0

if $gpu; then
    echo installing tensorflow-gpu
    pip install tensorflow-gpu==2.0
elif $cpu; then
    echo installing tensorflow
    pip install tensorflow==2.0
fi

echo installation complete!
