# Installation

## Dependencies

The following dependencies are required:

* Python (>=3.6)
* [Mujoco 1.50](https://www.roboti.us/index.html)

```sh
sudo apt-get install -y python3-pyqt5 protobuf-compiler libglew-dev patchelf
python3 -m pip install --user --upgrade -r requirements.txt
```

## Setup

1. Ensure that the dependencies listed above are statisfied.

1. Clone the repo:

```bash
git clone git@git.zlw-ima.rwth-aachen.de:dissertation_ennen/gps.git
cd gps
```

1. Compile protobuf:

```bash
./compile_proto.sh
```

1. Add to your `~/.bashrc`:

```bash
# Mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384  # Replace 384 with the actual driver version
# Required for GLEW, otheriwse it crashes at launch
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
```

   Run `source ~/.bashrc` or re-login to your shell.

## Docker

Alternatively, Docker can be used to run GPS.

1. Build the docker file:

```bash
docker build . -t gps
```

1. Run the container

```bash
docker run --name gps -v GPS_DIR/experiments:/gps/experiments gps gym_fetchreach_lqr
```
