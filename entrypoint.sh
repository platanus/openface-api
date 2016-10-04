#!/bin/bash

# In Docker, use a Bash login shell or source
# /root/torch/install/bin/torch-activate for the Torch environment
source /root/torch/install/bin/torch-activate

exec "$@"
