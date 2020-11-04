export CUGRAPH_HOME=$(pwd)

# Install NCCL
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev

conda install -y dlpack -c conda-forge
conda install -y rmm -c conda-forge

conda env create --name cugraph_dev --file conda/environments/cugraph_dev_cuda11.0.yml
conda activate cugraph_dev

./build.sh clean
./build.sh libcugraph
./build.sh cugraph

python test.py
