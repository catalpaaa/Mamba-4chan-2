mamba env create -f environment.yml
mamba activate mamba-4chan-2
mamba env config vars set CUDA_HOME=$CONDA_PREFIX
mamba activate mamba-4chan-2
git clone https://github.com/Dao-AILab/causal-conv1d
pip install -e causal-conv1d/
git clone https://github.com/state-spaces/mamba
pip install -e mamba/
git clone https://github.com/Dao-AILab/flash-attention
pip install -e flash-attention/
pip install git+https://github.com/pytorch/data