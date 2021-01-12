# Gamma-Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jannerm/gamma-models/blob/master/scripts/gamma-pendulum.ipynb)

Code release for [Gamma-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction](https://arxiv.org/abs/2010.14496).

## Run from your browser

For the quickest startup, we recommend running the notebook directly in your browser using [Google Colab](https://colab.research.google.com/github/jannerm/gamma-models/blob/master/scripts/gamma-pendulum.ipynb).

This notebook will generate a video that looks like the following:

<p align="center">
	<img src="https://people.eecs.berkeley.edu/~janner/gamma-models/blog/figures/gamma-colab.gif" width="80%">
</p>

The last cell in the notebook shows how the trained model can be used for value estimation:
```
visualize_values(prob_fn, n_steps=20)
```
as shown in Figure 4 of the paper. Increasing `n_steps` will improve the resolution of the value map but make the visualization take longer to generate.

## Run locally
1. Clone `gamma-models`
```
git clone https://github.com/jannerm/gamma-models.git
```
2. Create a conda environment and install `gamma`
```
cd gamma-models
conda env create -f environment.yml
conda activate gamma
pip install -e .
```
3. Add `gamma` as an IPython kernel and launch jupyter
```
python -m ipykernel install --user --name=gamma
jupyter notebook --port 6100 scripts
```
Open `gamma-pendulum-local.ipynb`, which matches the Colab notebook except for a bit of Colab-specific setup in the beginning.

## Reference

```
@inproceedings{janner2020gamma,
  title={$\gamma$-Models: Generative Temporal Difference Learning for Infinite-Horizon Prediction},
  author={Michael Janner and Igor Mordatch and Sergey Levine},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Acknowledgments
The underlying neural spline flow implementation is based on Andrej Karpathy's [python-normalizing-flows](https://github.com/karpathy/pytorch-normalizing-flows) repo, which in turn is based on Conor Durkan and Iain Murray's and [nsf](https://github.com/bayesiains/nsf) codebase.
