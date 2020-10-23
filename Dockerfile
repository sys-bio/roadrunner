# uses ciaranwelsh/ubuntu-base as a base. This image
# has a bunch of things that are common to several
# software packages that I'm testing on docker.
# Including:
#   - miniconda
#   - gcc 10.1, which is actively pointed to by the command gcc / g++
#   - cmake version 18, which is in $PATH
#   - several apt packages and an updated apt store.

FROM ciaranwelsh/ubuntu-base:latest

ENV DEBIAN_FRONTEND noninteractive
ENV PATH="/root/miniconda3/bin:$PATH"

# get the llvm dependency
# -X option for unzip supposedly
RUN wget https://github.com/sys-bio/LLVM6.0.1/releases/download/v0.0.1/LLVM6.0.1-ubuntu18.04-gcc10.1.zip \
    && unzip -X LLVM6.0.1-ubuntu18.04-gcc10.1.zip \
    && llvm_root="/LLVM6.0.1-ubuntu18.04-gcc10.1"

RUN git clone https://github.com/sys-bio/roadrunner.git \
    && cd roadrunner/
    && git checkout develop-new-buildsystem
    && mkdir build && cd build
    && cmake -DLLVM_INSTALL_PREFIX=$llvm_root -DCMAKE_INSTALL_PREFIX=../install ..
    && make


