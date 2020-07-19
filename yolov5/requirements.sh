cd /usr/local
sudo ln -snf /usr/local/cuda-10.1 cuda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
pip install tensorboard
pip install opencv-python
pip install matplotlib
pip install pyyaml
pip install scipy
pip install tqdm
conda install pytorch torchvision cudatoolkit=10.1

