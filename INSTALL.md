# Installation

## Dependencies

The following are required:

* python 2.7
* python-numpy (>=1.7.0)
* python-matplotlib (>=1.5.0)
* python-scipy (>=0.11.0)
* python-qt4
* ROS (kinetic)
* ros-kinetic-gazebo-ros-control and other packages

## Setup

1. Check that the dependencies above are installed.

2. Install boost and protobuf:
   ```bash
   sudo apt-get install libboost-all-dev libprotobuf-dev protobuf-compiler python-protobuf
   ```

3. Clone the repo:
   ```bash
   git clone git@git.zlw-ima.rwth-aachen.de:dissertation_ennen/gps.git
   cd gps
   ```

4. Compile protobuf:
   ```bash
   ./compile_proto.sh
   ```
5. Initialize the workspace and build the package:
   ```bash
   catkin_make
   ```
   The build will likely fail because of missing ROS packages, but that is ok for now.

6. Add to your `~/.bashrc`:
   ```bash
   source /path/to/gps/devel/setup.bash
   ```
   Run `source ~/.bashrc` or re-login to your shell.

7. If the build failed in step 5, install missing dependencies:
   ```bash
   rosdep install gps_agent_pkg
   ```
   and build the package again:
   ```bash
   catkin_make
   ```
