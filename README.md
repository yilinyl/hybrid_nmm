## :rainbow: A hybrid framework (neural mass model + ML) for SC-to-FC prediction

The current workflow simulates brain functional connectivity (FC) from structural connectivity (SC) with a neural mass model. Gradient descent is applied to optimize the parameters in the neural mass model. 

The pipeline contains the following components:
* Neural Mass Model (`models/torch_neural_mass.py`): It is an ODE system that describes the neural activities over time. The Wilson-Cowan model is implemented here with a connected network setting - each neural region is considered as a node in the brain network and connected via SC. The Wilson-Cowan model assumes each node contains two types of neural populations: the excitatory and inhibitory cells. The definition can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1484078/) and [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3139941/).
* Hemodynamic Model (`models/hrf_torch.py`): This module down samples and transforms the neural activities into Blood Oxygen Level Dependence (BOLD) signals. The code is adapted from [the Virtual Brain](https://github.com/the-virtual-brain/tvb-root) implementation of [the Balloon model](https://www.fil.ion.ucl.ac.uk/~karl/Nonlinear%20Responses%20in%20fMRI%20The%20Balloon%20Model.pdf).

### Requirement

PyTorch (my version is 1.10.0)

### Usage

An example of running the pipeline can be found at `run.sh`. 
Please update path to your data.
