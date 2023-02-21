# This dockerfile contains commands for a base environment suitable for
# building manylinux pip wheels.
# The default gcc/g++ is 9.3
# Conda is used to manage Python versions.
#
# This dockerfile installs roadrunner dependencies and llvm package
# But does not build roadrunner, which is the subject of another dockerfile

FROM sysbiouw/roadrunner-manylinux2014-base

#RUN git clone https://github.com/sys-bio/llvm-13.x.git \
#    && cd llvm-13.x
#RUN cd /llvm-13.x/llvm && mkdir build && cd build \
#    && cmake -DCMAKE_INSTALL_PREFIX=/install-llvm-13.x -DCMAKE_BUILD_TYPE=Release ..
#RUN cd /llvm-13.x/llvm/build \
#    && cmake --build . --target install --config Release -j 12
#RUN rm -rf /llvm-13.x
#
#RUN git clone https://github.com/sys-bio/libroadrunner-deps.git --recurse-submodules
#RUN cd libroadrunner-deps && mkdir build && cd build \
#    && cmake -DCMAKE_INSTALL_PREFIX=../install-libroadrunner-deps -DCMAKE_BUILD_TYPE=Release .. \
#    && cmake --build . --target install --config Release -j 12

#Run above to re-create build; for now, just use pre-built:

RUN wget https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm-13.x-manylinux2014.tgz \
    && tar -xvf llvm-13.x-manylinux2014.tgz \
    && rm llvm-13.x-manylinux2014.tgz

RUN wget https://github.com/sys-bio/libroadrunner-deps/releases/download/v2.1/libroadrunner-deps-manylinux2014-Release.zip \
    && unzip libroadrunner-deps-manylinux2014-Release.zip \
    && rm libroadrunner-deps-manylinux2014-Release.zip \
    && mv libroadrunner-deps-manylinux2014-Release/install-azure/Release install-libroadrunner-deps


