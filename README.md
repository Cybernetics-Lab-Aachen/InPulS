GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

See INSTALL.md for new catkin installation instructions.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.

## Working with a remote machine

1. Both machines have to be able to find each other by hostname. An easy way to achieve this is to enter the hostnames in the file `/etc/hosts` on both machines.

1. Run `roscore` on one of the machines. On both machines, run `export ROS_MASTER_URI=http://<hostname>:1131`. Replace `<hostname>` with the hostname of the machine running `roscore`. (This has to be run in every new terminal session; alternatively it can be written to `~/.bashrc` or any other shell configuration script)

1. To run a graphical application on a remote machine via ssh, X11 forwarding has to be activated. Run `ssh -X <hostname>` instead of `ssh <hostname>`.

1. Sources:
   * http://wiki.ros.org/ROS/Tutorials/MultipleMachines
