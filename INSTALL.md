###Installation###

##Dependencies##

The following are required

* python 2.7
* numpy (>=1.7.0)
* matplotlib (>=1.5.0)
* scipy (>=0.11.0)
* ROS (kinetic)

##Setup##

1. Check that the dependencies above are installed

2. Install boost and protobuf:

   ```bash
   sudo apt-get install libboost-all libprotobuf-dev protobuf-compiler python-protobuf
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

5. Add to your `~/.bashrc`:

   ```bash
   source /path/to/gps/devel/setup.bash
   ```

   Re-login to your shell or run `source ~/.bashrc`

6. Compile the package:

   ```bash
   catkin_make
   ```