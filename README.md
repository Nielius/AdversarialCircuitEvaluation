# Adversarial Circuit Evaluation

This is the code for the paper "Adversarial Circuit Evaluation" ([openreview](https://openreview.net/forum?id=I5E9ZZNBjT)).
It builds on <https://github.com/ArthurConmy/Automatic-Circuit-Discovery>.

The code that is relevant to adversarial circuit evaluation is in `acdc/new/adv_opt`,
which directory contains all the experiments.
If you're interested in looking at the technical details
of how we run resample ablations, look at the class `acdc.new.adv_opt.masked_runner.MaskedRunner`,
which builds on our adapation of edge-level subnetwork probing (see `subnetwork_probing.masked_transformer.EdgeLevelMaskedTransformer`).


## Installation:

First, install the system dependencies for either [Mac](#apple-mac-os-x) or [Linux](#penguin-ubuntu-linux).

Then, you need Python 3.8+ and [Poetry](https://python-poetry.org/docs/) to install ACDC, like so

```bash
git clone git+https://github.com/AlignmentResearch/acdc.git
cd acdc
poetry env use 3.10      # Or be inside a conda or venv environment
                         # Python 3.10 is recommended but use any Python version >= 3.8
poetry install
```

### System Dependencies

#### :penguin: Ubuntu Linux

```bash
sudo apt-get update && sudo apt-get install libgl1-mesa-glx graphviz build-essential graphviz-dev
```

You may also need `apt-get install python3.x-dev` where `x` is your Python version (also see [the issue](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/issues/57) and [pygraphviz installation troubleshooting](https://pygraphviz.github.io/documentation/stable/install.html))

#### :apple: Mac OS X

On Mac, you need to let pip (inside poetry) know about the path to the Graphviz libraries.

```
brew install graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
```
