# This dockerfile contains commands for a base environment suitable for
# building manylinux pip wheels.
# The default gcc/g++ is 9.3
# Conda is used to manage Python versions.
#
# This dockerfile installs roadrunner dependencies and llvm package
# But does not build roadrunner, which is the subject of another dockerfile

FROM sysbiouw/roadrunner-manylinux2014-add-deps

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /Miniconda3

RUN /Miniconda3/bin/conda create -y --name py39 python=3.9 pytest
RUN /Miniconda3/bin/conda create -y --name py310 python=3.10 pytest
RUN /Miniconda3/bin/conda create -y --name py311 python=3.11 pytest -c conda-forge
RUN /Miniconda3/bin/conda create -y --name py312 python=3.12 pytest -c conda-forge

RUN /Miniconda3/bin/conda init && bash ~/.bashrc && . ~/.bashrc
