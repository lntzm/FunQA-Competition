ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

WORKDIR /workspace

COPY ./test.sh ./test.sh
COPY ./train.sh ./train.sh

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-dev\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip install
RUN pip install --no-cache-dir --upgrade pip wheel setuptools -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install --no-cache-dir h5py termcolor scipy matplotlib nltk tqdm pandas pillow -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install --no-cache-dir openai einops fvcore SentencePiece accelerate bitsandbytes -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install --no-cache-dir \
validators==0.20.0 \
lmdb==1.2.1 \
nltk==3.6.2 \
natsort==7.1.1 \
wand==0.6.7 \
strsimpy==0.2.1 \
omegaconf==2.3.0 \
pandas \
iopath==0.1.10 \
timm==0.6.13 \
opencv-python==4.7.0.72 \
decord==0.6.0 \
webdataset==0.2.48 \
transformers==4.28.0 \
-i https://pypi.mirrors.ustc.edu.cn/simple/

# export conda in shell
RUN echo '\n' >> /root/.bashrc
RUN echo '# activate conda' >> /root/.bashrc
RUN echo 'source /opt/conda/bin/activate base' >> /root/.bashrc

# conda clean
RUN conda clean --all