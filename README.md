# Uncertainty-aware spot rejection rate as quality metric for proton therapy

Alexander Schilling on behalf of the
[Bergen pCT collaboration](https://www.uib.no/en/ift/142356/medical-physics-bergen-pct-project)
and the [SIVERT research training group](https://sivert.info/)

Implementation of the paper "Uncertainty-aware spot rejection rate as quality metric for proton therapy."

## Repository Structure

`src` contains the Python source code. Executable scripts are at the top level and all modules are in sub-directories.

`data` contains supplementary data and the directory is used as output of the scripts.

Everything is implemented in Python with the required 3rd-party libraries listed in requirements.txt, which can be
installed via pip:

    $ pip install -r requirements.txt

## Simulation Data

36258 treatment spot simulations in an anthropomorphic head phantom (715-HN) (Giacometti et al. 2017)
and 35673 simulations in a second anthropomorphic head phantom (VHF) (Ackermann et al. 1995)
with 1e7 primary protons each have been carried out.
Used spot/rotation/energy combinations were determined through probing simulations,
the results of which can be found in `data/probing_results.csv` and `data/probing_results_vhf.csv`.
The simulation data is publicly available:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8192778.svg)](https://doi.org/10.5281/zenodo.8192778)

## Feature Generation

> Feature generation requires a MetaImage file of the head phantom you want to use
> and a Singularity image of the simulation environment (for lateral shift simulation).

The raw simulation data can be used to generate features for machine learning through the `generate_features.py` script:

    $ cd src
    $ python generate_features.py -o ../data/features.csv "../data/sims/*/*.json"

By default, this uses the 715-HN phantom at `data/imageDump.mhd`. `--phantom` and `--rsp-file` can be used to change
these defaults.

By specifying shift parameters (e.g., `-s 5`), the script generates features for cases with errors in the form of
lateral shifts from 1 mm up to the given value in 1 mm intervals.

The resulting feature sets (train, validation, test, and shifts) are publicly available:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8192513.svg)](https://doi.org/10.5281/zenodo.8192513)

## Range Verification

To run a full evaluation of the range verification method on the above features, run the `eval.py` script:

    $ cd src
    $ python eval.py -c config/example.json -w ../data/eval -d cuda:0

The parameters of the run are defined in the configuration file at `src/config/example.json`. This should be adjusted
to point to the correct feature files. The rest of the parameters are set to reproduce the paper.

## Timing

To measure the runtime of different parts of the implementation, use `timing.py`.

This requires a fully trained model in a given working directory and a number of input simulation metafiles.

    $ cd src
    $ python timing.py -c config/example.json -w ../data/eval -d cuda:0 ../data/sims/*/*.json

## References

Ackerman M. J., Spitzer V. M., Scherzinger A. L., and Whitlock D. G., 1995.
The Visible Human data set: an image resource for anatomical visualization
Medinfo 8 Pt 2, pp. 1195–1198.

Giacometti, V., Guatelli, S., Bazalova-Carter, M., Rosenfeld, A. B. and Schulte, R. W., 2017.
Development of a high resolution voxelised head phantom for medical physics applications.
Physica Medica, 33, pp.182-188.
