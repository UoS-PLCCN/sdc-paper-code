# Sampled-Data Control for PBCNs

Control for PBCNs using Sampled-Data methods. In essence, dynamic action repetition for control of PB(C)Ns.

## How to run

1. Install `gym-PBN`:

    ```sh
    python -m pip install -e ../gym-PBN
    ```

2. Install PyTorch. Find the command to do this for your CUDA version [here](https://pytorch.org/get-started/locally/).
3. Simply run `deep_q_learning.py` for Deep Q Learning or `q_learning.py` for classical Q learning.

## Environment Setup

To set up the environment you need to provide `logic_func_data` in the `gym.make` command. This should be a **Tuple** of:

1. An array containing the string names of all the nodes. Control nodes need to be first in PBCNs.
2. An array of arrays. Each inner array should contain the logic functions associated with a node. The index of this array in the outer array should match the node's index in the array of node names. Each logic function is a tuple of a logic expression (the function) using node names as literals and pythonic logic operators, and the associated probability of that function activating.

You may also want to provide the actual control target using `goal_config`. If you don't, the last attractor out of the computed attractors will be picked as the target. The goal config needs two keys:

1. `"all-attractors"`: list of sets of tuples of attractor states. Listing out all the attractors, essentially
2. `"target"`: set of tuples of attractor states for the target attractor.
