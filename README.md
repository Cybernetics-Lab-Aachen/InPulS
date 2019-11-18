# GPS

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization.

This implementation is loosely based on a code provided by [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps) and was enhanced here for industrial usage. This includes algorithm and policy improvements, support for OPC-UA communication and multiple vizualisation tools.

This project is funded by VDMA germany. For further information please visit [i40-inpuls.de](http://i40-inpuls.de)

## Installation

See INSTALL.md for installation instructions.

## Run Experiment

```bash
python .\main.py gym_fetchreach_lqr
```

What to see what the robot is doing?

Set `'render'` to `True` in the agent block in `experiments\gym_fetchreach_lqr\hyperparams.py`.
