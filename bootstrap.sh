# Version checks
# This shell requires CUDA10.0 + cuDNN 7.6 + Python 3.6 + (and it will install Tensorflow 1.13.1 + Keras 2.2.4 for you)
# CUDA
cat /usr/local/cuda/version.txt

# cuDNN
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# For tf&CUDA&cuDNN&Keras
# see:https://wiki.imalan.cn/posts/TensorFlow%E3%80%81Keras%E3%80%81CuDNN-%E4%B9%8B%E9%97%B4%E7%9A%84%E7%89%88%E6%9C%AC%E8%A6%81%E6%B1%82/

mkdir -p ~/envs
cd ~/envs
virtualenv -p python3.6 --no-site-packages keras2
source keras2/bin/activate
pip install -r requirements.txt
python testtf.py

# DEBUGS
# matplotlib 在 3 版本后要求必须 3.5以上的python才能安装
# scikit-image 要求 matplotlib >=2.0.0 && != 3.0.0
