#!/bin/bash

python $ACDC_ROOT_DIR/acdc/nudb/adv_opt/main.py num_epochs=10000 use_wandb=True wandb_run_name=reverse_10000 random_seed=4321
