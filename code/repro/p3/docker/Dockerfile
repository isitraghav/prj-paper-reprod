# P3 SVM with the paper's OpenCV version (3.3.1 + contrib 3.3.1; no wheels exist any more, so built from source).
# Python 3.6 (Ubuntu 18.04) with the same scikit-learn / numpy / joblib as env p3svm (code/repro/env/p3svm).
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake wget ca-certificates \
      python3-dev python3-pip pkg-config libjpeg-dev libpng-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir pip==21.3.1 && python3 -m pip install --no-cache-dir numpy==1.18.5 \
      scipy==1.4.1 scikit-learn==0.22.2.post1 joblib==0.14.1
WORKDIR /build
RUN wget -q https://github.com/opencv/opencv/archive/3.3.1.tar.gz -O cv.tgz && tar xzf cv.tgz && \
    wget -q https://github.com/opencv/opencv_contrib/archive/3.3.1.tar.gz -O contrib.tgz && tar xzf contrib.tgz && \
    mkdir opencv-3.3.1/build && cd opencv-3.3.1/build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/build/opencv_contrib-3.3.1/modules \
      -D BUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,flann,xfeatures2d,python3 \
      -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF -D WITH_CUDA=OFF \
      -D WITH_GTK=OFF -D WITH_FFMPEG=OFF -D WITH_IPP=OFF -D PYTHON3_EXECUTABLE=/usr/bin/python3 .. && \
    make -j6 && make install && ldconfig && cd / && rm -rf /build
RUN python3 -c "import cv2; print(cv2.__version__); cv2.KAZE_create(); cv2.xfeatures2d.VGG_create()"
WORKDIR /repo
