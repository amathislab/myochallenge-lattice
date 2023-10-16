FROM nvcr.io/nvidia/pytorch:21.09-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update -q\
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs \
libgl1-mesa-dev \
libgl1-mesa-glx \
libglew-dev \
libosmesa6-dev \
software-properties-common \
patchelf

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.jsondf 
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN useradd --create-home gymuser
USER gymuser
ENV PATH="/home/gymuser/.local/bin:$PATH"

COPY --chown=gymuser docker/requirements.txt /usr/share/requirements.txt
COPY docker/requirements.txt /usr/share/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /usr/share/requirements.txt

# WORKDIR /opt/isaacgym

# copy gym repo to docker
# COPY --chown=gymuser docker/isaacgym .
# COPY docker/isaacgym .

# install gym modules
# RUN cd python && pip install -q -e .

# ADD docker/isaacgym /usr/share/isaacgym
# RUN cd /usr/share/isaacgym/python && pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

RUN pip install myosuite==1.7.1
# RUN pip install --upgrade pickle5==0.0.11 cloudpickle==2.2.1
RUN python3 -c "import mujoco"
# RUN python3 -c "import mujoco_py"

RUN mkdir /home/gymuser/data
RUN mkdir /home/gymuser/output
RUN mkdir /home/gymuser/src

COPY data /home/gymuser/data
COPY output /home/gymuser/output
COPY src /home/gymuser/src

CMD [ "python", "/home/gymuser/src/agent_mani_phase1.py" ]
