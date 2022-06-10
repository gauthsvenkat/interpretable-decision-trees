ARG PARENT_IMAGE
FROM $PARENT_IMAGE

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# Install system dependencies
RUN apt-get update && apt-get -y install \
    g++ \
    cmake \
    libopenmpi-dev \
    libz-dev \
    xvfb \
    python3-opencv \
    graphviz \
    libffi-dev \
    locales \
    ffmpeg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN \
    pip install stable-baselines[mpi]==2.10.0 \
    box2d-py==2.3.5 \
    pybullet \
    gym-minigrid \
    scikit-optimize \
    optuna \
    pytablewriter \
    pydotplus \
    protobuf==3.4.0 \
    tensorflow==1.8.0 \
    jupyter \
    interpretableai \
    && rm -rf $HOME/.cache/pip

# python 3.5 has a weird problem with f string so fix it 
RUN sed -i 's/f"{self.src_path} is not mounted successfully"/"{} is not mounted successfully".format(self.src_path)/' /usr/local/lib/python3.5/site-packages/interpretableai/installation.py
# Install julia and system image
RUN python -c "import interpretableai; interpretableai.install_julia(); interpretableai.install_system_image()"
RUN python -c "from interpretableai import iai;"

COPY docker/entrypoint.sh /tmp/
COPY iai-config/iai.lic /root/iai.lic
ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash