# Understanding approximate and unrolled dictionary learning for pattern recovery

This package is intended to reproduce the code of the following paper: 

Benoît Malézieux, Thomas Moreau, Matthieu Kowalski. Understanding approximate and unrolled dictionary learning for pattern recovery. ICLR 2022.

## installation
This package can be installed using **pip** with the following command:
```
pip install -e .
```

WARNING: You should install alphacsc separately from https://github.com/alphacsc/alphacsc  by following the instructions on https://alphacsc.github.io to make it work. Moreover, the size of MNE data downloaded in experiments on MEG signals is 1.5GB.

## content
* main code in **plipy**
* experiments and figures in **experiments_approximate**
* results in experiments_approximate/{results,figures}
* implementation of convolutional k-svd from Yellin et al. (2017) in ksvd_expe

## figures

Figures can be reproduced with the following files contained in experiments_approximate:

* figure 1, 2: 
	* experiments/jacobian.py
	* figures_generation/jacobian_fig.py


* figure 3, C:
	* experiments/gradients.py
	* figures_generation/gradient_fig.py


* figure 4:
	* experiments/optim_path.py
	* figures_generation/optim_path_fig.py


* figure 5:
	* experiments/optim_image.py ; experiments/dico_minima.py
	* figures_generations/optim_image.py ; figures_generation/boxplot_psnr_noise.py ; figures_generation/landscape_1D.py


* figure 6:
	* experiments/sto_path.py
	* figures_generation/optim_sto.py


* figure 7:
	* examples/example_meg.py
	* examples/figure_meg.py


* figure 8:
	* experiments/alpha_csc_atoms.py ; experiments/create_dico_alphacsc ; experiments/benchmark_meg.py


* figure D:
	* experiments/sto_path.py
	* figures_generation/optim_sto_hist.py


* figure E:
	* experiments/sto_path_online.py
	* figures_generation/sto_online_figure.py


Generating figures may take a long time and a large GPU usage. The experiments have been run on GPU NVIDIA Tesla V100-DGXS 32GB in:

* figure 1: < 1h (no GPU)
* figure 2: < 1h (no GPU)
* figure 3: < 1h (GPU)
* figure 4: < a week (GPU)
* figure 5: < 2h (GPU)
* figure 6: < 4h (GPU)
* figure 7: < 10min (GPU)
* table 1: < 48h (GPU)
* figure B: < 5min (GPU)
* figure C: < 1h (GPU)
* figure D: < 4h (GPU)
* figure E: < 6h (GPU)

All classes in plipy integrate a parameter ```device``` for GPU configuration, except for ```DL``` in dl_sklearn.py.


