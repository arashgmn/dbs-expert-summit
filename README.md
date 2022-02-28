
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arashgmn/dbs-expert-summit/HEAD)

# 2nd DBS Expert Summit
scripts for the poster result reproduction. 

# Dependencies
You need a python (3.x) environment with the following modules installed:
- numpy
- scipy
- [sdeint](https://github.com/mattja/sdeint): Stochastic systems for noisy inputs are implemented, but we have not used them in this work. Thus, can be commented out in the source code.
- matplotlib
- seaborn (optional)
- jupyter (optional)

# How to execute
You can proceed with either of the following:

- **Recommended**: Interactively execute the jupyter notebook located at `scripts/summit.ipynb` using binder (click on "launch binder" badge above). No installation is required.  Note that building the binder image and launching the server may take up to 2 minutes.
- Read the notebook as an HTML from `scripts/summit.html`. No installation is required but you may need to download the html file and open it in your browser if Github doesn't load it.
- **Not Recommended**: First, make a python environment with the dependencies mentioned above. To do so, simply clone the repo, navigate in, and execute `pip install -r requirements.txt` in your terminal. Then, run either `scripts/summit.py` script or open `summit.ipynb` notebook. *This method installs packages on your default python environment, which is likely undesireable*.
