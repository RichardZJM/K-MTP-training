# A Collection of the Important Commands for MTP Usage

This document outlines several important commands which are niche (unique to the MTP) although they are often used when dealing with MTP.

### Passive Training of an MTP

This initiates the passive training of an MTP, whether untrained or existing with respect to a specified set of training configurations. Most of the arguments concern the relative importances of properties in the loss function or parameters in the optimization. These generally do not need to be modified.

Generic:

```sh
<mlp binary> train <existing or untrained potential> <configuration  file> --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=<output name>
```

CAC Example:

```sh
/global/home/hpc5146/mlip-2/bin/mlp train 08.mtp train.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=pot.mtp
```

srun --account=def-hpcg1725 --cpus-per-task=4 --mem-per-cpu=4G --partition=reserved --qos=privileged --pty bash -l
