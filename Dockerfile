FROM python:3.6

RUN apt-get update && apt-get install -y \
	# Protobuf
	protobuf-compiler \
	# Dependencies for Mujoco
	libglew-dev patchelf libosmesa6-dev libglfw3-dev

# Install Mujoco
RUN mkdir -p /root/.mujoco \
	&& wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
	&& unzip mujoco.zip -d /root/.mujoco \
	&& rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}

# Change work dir to gps location
WORKDIR /gps/

# Install python dependencies
COPY ./requirements.txt requirements.txt
RUN	python3 -m pip install -U --user -r requirements.txt

# Compile protobuf files
COPY compile_proto.sh /gps/
COPY proto/ /gps/proto/
RUN	./compile_proto.sh

# Add source
COPY main.py /gps/
COPY gps/ /gps/gps/

VOLUME /gps/experiments/

ENTRYPOINT ["python3", "main.py"]
