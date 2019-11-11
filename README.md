# RLIF

This package contains:
1. A reinforcement learning based algorithm for RNA inverse folding.
2. A graphical interface for RNA design.
3. A command line interface for generating solutions for RNA secondary structures.

# Requirements:

1. Conda
2. ViennaRNA
3. Python==3.7
4. pip:
   1. Base: 
      1. numpy==1.17.1
      2. tensorflow==1.13.1
      3. stable-baselines==2.7.0
      4. pyyaml==3.13
      5. gym
      6. forgi
      7. tqdm
   
   2. GUI version:
      1.  PyQt5
      2.  pyqtgraph
      3.  matplotlib

# Installation:

```
git clone https://github.com/andriusbern/rlif
cd rlif
conda env create -f rlif.yml
conda activate rlif
pip install -e .
```

# Usage

For the graphical user interface:

```
python rlif/GUI.py
```

For the command line interface:

```
python rlif/CLI.py
```



