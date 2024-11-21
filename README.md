# dm Stiefel

[![DOI](https://zenodo.org/badge/796548095.svg)](https://zenodo.org/doi/10.5281/zenodo.12540300)

This is the repository for the paper "Unified Framework for Calculating Convex Roof Resource Measures". [arxiv-link](https://arxiv.org/abs/2406.19683)

🚀 Exciting News! We've launched the `numqi` package [github/numqi](https://github.com/numqi/numqi), combining all the functionalities of this repository and even more! 🌟 To dive into these features, just install `numqi` using `pip install numqi`, and explore the relevant functions within the package. 🛠️

## quick start

This project is based on `numqi` package [github-link](https://github.com/numqi/numqi), which is a package for numerical optimization for quantum information. To install `numqi`, run the following command (see [numqi-doc](https://numqi.github.io/numqi/installation/) for detailed installation instruction):

```bash
pip install numqi
```

Then, clone this repository and cd into the directory

```bash
git clone git@github.com:numqi/dm-stiefel.git
cd dm-stiefel
```

To verify the installation, run the following command

```bash
pytest
```

and you should see all tests passed like below:

```bash
> pytest
============================ test session starts =================
platform linux -- Python 3.12.3, pytest-8.2.0, pluggy-1.5.0
rootdir: ~/project/dm-stiefel
collected 7 items

test_utils.py .......                                       [100%]
==================================================================
```

then you are ready to go! please try the jupyter notebook `paper_data.ipynb` to reproduce the data in the paper. Some figures are generated in the script file `draft_fig00.py`.

There are more script files for various experiments in the paper.

1. `draft_coherence.py`: geometric measure of coherence, coherence of formation
2. `draft_magic.py`: stabilizer entropy
3. `draft_linear_entropy.py`: linear entropy of entanglement
4. `draft_gme.py`: geometric measure of entanglement
5. `draft_3tangle.py`: 3-tangle
6. `draft_misc.py`: miscellaneous

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
