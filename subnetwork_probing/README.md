# Setup

This implementation of Subnetwork Probing should install by default when installing the ACDC code.

We no longer use a fork of `transformer_lens`, and instead implement masking with a `MaskedTransformer` wrapper class around the model.

Most of the complexity of node SP is in `train.py`.

Edge SP was implemented by Thomas and lives in `train_edge_sp.py` and `sp_utils.py`. In edge SP, we overwrite the inputs to nodes rather than their outputs. We keep track of the residual stream component, so we can mask them separately; non-ablated activations come from `forward_cache` and ablated activations from `ablation_cache`

# Subnetwork Probing

[Low-Complexity Probing via Finding Subnetworks](https://github.com/stevenxcao/subnetwork-probing)  
Steven Cao, Victor Sanh, Alexander M. Rush  
NAACL-HLT 2021  

# HISP 

[Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) Michel et al 2019
