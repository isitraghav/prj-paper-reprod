# P3 CNN on TensorFlow 2.12, GPU only. TF >= 2.11 has no native-Windows GPU build, so it runs in the official
# TF 2.12 GPU image (CUDA 11.8, cuDNN 8.6, Python 3.8) under Docker Desktop (WSL2) with --gpus all.
# Same packages as env p3cnn (code/repro/env/p3cnn) apart from TensorFlow itself.
FROM tensorflow/tensorflow:2.12.0-gpu
RUN python3 -m pip install --no-cache-dir numpy==1.23.5 pillow==9.5.0 pandas==2.0.3 scikit-learn==1.3.2 matplotlib==3.7.5
RUN python3 -c "import tensorflow as tf; assert tf.__version__ == '2.12.0', tf.__version__; print(tf.__version__, tf.sysconfig.get_build_info()['cuda_version'], tf.sysconfig.get_build_info()['cudnn_version'])"
WORKDIR /repo
