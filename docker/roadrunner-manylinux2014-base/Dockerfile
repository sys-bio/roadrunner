# This dockerfile contains commands for a base environment suitable for
# building manylinux pip wheels.
# The default gcc/g++ is 9.3
# Conda is used to manage Python versions.
#
# This dockerfile installs roadrunner dependencies and llvm package
# But does not build roadrunner, which is the subject of another dockerfile

FROM quay.io/pypa/manylinux2014_x86_64
RUN yum install -y wget

# build the latest gcc (11.1.0), which should keep us going for a while
# instructions: https://linuxhostsupport.com/blog/how-to-install-gcc-on-centos-7/
RUN wget http://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-11.1.0/gcc-11.1.0.tar.gz
RUN tar zxf gcc-11.1.0.tar.gz && yum -y install bzip2
RUN cd gcc-11.1.0 && ./contrib/download_prerequisites
RUN cd gcc-11.1.0 && mkdir build && cd build && ../configure --prefix=/usr/local/gcc-11.1.0 --disable-multilib
RUN cd gcc-11.1.0/build && make -j 12 CPPFLAGS=-O3 CXXFLAGS=-O3
RUN cd gcc-11.1.0/build && make install
RUN cd .. && rm -rf gcc-11.1.0 gcc-11.1.0.tar.gz

#RUN yum install -y nano

RUN yum install -y pcre-devel.x86_64

ENV swig_source_dir /swig-4.0.2
ENV swig_install_dir /swig-4.0.2/install-swig-4.0.2
RUN  mkdir -p $swig_source_dir && wget -q --no-check-certificate https://netcologne.dl.sourceforge.net/project/swig/swig/swig-4.0.2/swig-4.0.2.tar.gz
RUN  tar -xf swig-4.0.2.tar.gz
RUN  cd $swig_source_dir && ./configure --prefix=$swig_install_dir
RUN  cd $swig_source_dir && make -j 12 && make install
RUN  cd $swig_source_dir && mv $swig_install_dir .. && rm -r -f * && mv ../install-swig* .
RUN  rm -r -f swig-4.0.2.tar.gz

#RUN git config --global core.editor nano && git config --global pager.branch false


#RUN echo -e "export PATH=\"/usr/local/gcc-11.1.0/bin:$PATH\"" >> ~/.bashrc \
#    && echo -e "export LD_LIBRARY_PATH=\"/usr/local/gcc-11.1.0/lib:$LD_LIBRARY_PATH\"" >> ~/.bashrc

# some additional commands to enable docker integration with CLION
# which is a helpful tool for debugging manylinux builds
# see https://www.jetbrains.com/help/clion/docker.html
# and https://blog.jetbrains.com/clion/2020/01/using-docker-with-clion/#using-the-remote-development-workflow-with-docker
RUN yum install -y openssh-server openssh-clients
RUN ssh-keygen -A

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/libexec/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion

RUN useradd -m user \
  && yes password | passwd user
RUN yum install -y initscripts

RUN yum install -y qt5-qtbase-devel check
