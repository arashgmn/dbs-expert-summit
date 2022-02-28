# 2nd DBS Expert Summit
scripts for the poster result reproduction. 

# How to run
You can run either the `summit.py` or`summit.ipynb` from the `/scripts` directory. Alternatively, you may view the executed jupyter notebook, as an HTML file from [here](https://github.com/arashgmn/dbs-expert-summit/blob/main/scripts/summit.html) or run the notebook interactively without any local installtions using the  binder image (click on the badge below):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arashgmn/dbs-expert-summit/HEAD)

**Note**: Building the binder image and launching the server may take up to 1-2 minutes. After the server launch navigate to `scripts/` and open `summit.ipynb`.

# Dependencies
You need a python (3.x) environment with the following modules installed:
- numpy
- scipy
- matplotlib
- seaborn (optional)
- [sdeint](https://github.com/mattja/sdeint): Stochastic systems for noisy inputs are implemented, but we have not used them in this work. Thus, can be commented out in the source code.
