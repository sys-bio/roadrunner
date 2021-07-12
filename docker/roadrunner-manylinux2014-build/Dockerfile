# This dockerfile contains commands for a base environment suitable for
# building manylinux pip wheels.
# The default gcc/g++ is 9.3
# Conda is used to manage Python versions.
#
# This dockerfile installs roadrunner dependencies and llvm package
# But does not build roadrunner, which is the subject of another dockerfile
#
# When looking for bugs in the many linux build:
#   1) Ensure you have the docker dependencies image:
#       - docker pull ciaranwelsh/roadrunner-manylinux2014-base:latest
#   2) Change git branch to the one you want to build (the one where your tests are failing):
#       - RUN cd $roadrunnerSource && git pull && git checkout expand-unittests
#   3) Run docker build . inside the "docker/roadrunner-manylinux2014-build" folder
#



FROM ciaranwelsh/roadrunner-manylinux2014-base

ENV conda /Miniconda3/bin/conda
ENV bashrc /root/.bashrc
ENV roadrunnerSource /roadrunner
ENV roadrunnerBuild37 /roadrunner/build37
ENV roadrunnerBuild38 /roadrunner/build38
ENV roadrunnerBuild39 /roadrunner/build39
ENV roadrunnerInstall37 /roadrunner/install37
ENV roadrunnerInstall38 /roadrunner/install38
ENV roadrunnerInstall39 /roadrunner/install39
ENV swigExec /swig-4.0.2/install-swig-4.0.2/bin/swig

# install numpy in all conda envs
RUN $conda init && . $bashrc \
    && conda activate py37 \
    && pip install numpy \
    && conda activate py38 \
    && pip install numpy \
    && conda activate py39 \
    && pip install numpy

# install a dependency
RUN yum install -y ncurses-devel

# get roadruner source, checkout kinsol branch which has minor fix for building on centos
# note, build master or develop once temporary kinsol branch is merged.
RUN pwd
RUN git clone https://github.com/sys-bio/roadrunner.git $roadrunnerSource

RUN cd $roadrunnerSource && git pull && git checkout expand-unittests

## build against python 3.7
RUN $conda init && . $bashrc && conda activate py37 && cd $roadrunnerSource \
    && mkdir -p $roadrunnerBuild37 && cd $roadrunnerBuild37 \
    && cmake -DLLVM_INSTALL_PREFIX=/llvm-6.x/install-llvm-6.x -DRR_DEPENDENCIES_INSTALL_PREFIX=/libroadrunner-deps/install-libroadrunner-deps -DCMAKE_INSTALL_PREFIX=$roadrunnerInstall37 -DBUILD_PYTHON=ON -DBUILD_RR_PLUGINS=ON -DBUILD_TESTS=ON -DPython_ROOT_DIR=/Miniconda3/envs/py37 -DSWIG_EXECUTABLE=$swigExec ..
RUN cd $roadrunnerBuild37 \
    && cmake --build . --target install --config Release -j 1
#RUN $conda init && . $bashrc && cd $roadrunnerBuild37 \
#    && ctest --verbose --extra-verbose
#
## build against python 3.8
#RUN $conda init && . $bashrc && conda activate py38 && cd $roadrunnerSource \
#    && mkdir -p $roadrunnerBuild38 && cd $roadrunnerBuild38 \
#    && cmake -DLLVM_INSTALL_PREFIX=/llvm-project/llvm/install-llvm-6.x-rel -DRR_DEPENDENCIES_INSTALL_PREFIX=/libroadrunner-deps/install-libroadrunner-deps -DCMAKE_INSTALL_PREFIX=$roadrunnerInstall38 -DBUILD_PYTHON=ON -DBUILD_RR_PLUGINS=ON -DBUILD_TESTS=ON -DPython_ROOT_DIR=/Miniconda3/envs/py38 ..
#RUN cd $roadrunnerBuild38 \
#    && cmake --build . --target install --config Release -j 12
#RUN $conda init && . $bashrc && cd $roadrunnerBuild38 \
#    && ctest --verbose --extra-verbose
#
#
## build against python 3.9
#RUN $conda init && . $bashrc && conda activate py39 && cd $roadrunnerSource \
#    && mkdir -p $roadrunnerBuild39 && cd $roadrunnerBuild39 \
#    && cmake -DLLVM_INSTALL_PREFIX=/llvm-project/llvm/install-llvm-6.x-rel -DRR_DEPENDENCIES_INSTALL_PREFIX=/libroadrunner-deps/install-libroadrunner-deps -DCMAKE_INSTALL_PREFIX=$roadrunnerInstall39 -DBUILD_PYTHON=ON -DBUILD_RR_PLUGINS=ON -DBUILD_TESTS=ON -DPython_ROOT_DIR=/Miniconda3/envs/py39 ..
#RUN cd $roadrunnerBuild39 \
#    && cmake --build . --target install --config Release -j 12
#RUN $conda init && . $bashrc && cd $roadrunnerBuild39 \
#    && ctest --verbose --extra-verbose
#
#











