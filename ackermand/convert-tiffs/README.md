# Relabel components based on connectivity
Amira can only handle 8-bit segmentations, but often we have more objects than 255. So we want to relabel them into the 1-255 space, while ensuring that any two that are touching have different ids. To do this we first create a graph of touching ids and then relabel that. `graph_recoloring.py` is for this purpose.

## Installation
To install, first download `graph_recoloring.py` and `graph_recoloring_environment.yaml`. Next, perform the following steps:

1. Create a new conda environment: `conda create -n myenv python=3.11`
2. Activate the environment: `conda activate myenv`
3. Update the environment with the required dependencies: `conda env update --file graph_recoloring_environment.yml`

## Running
To run the code, activate your environment if not activated already:  `conda activate myenv`. Then you can run `python path/to/graph_recoloring.py -i /path/to/image/you/want/to/relabel.tif -o /path/to/output_directory`. The `output_directory` will contain the output image as well as a `pkl` file containging the set of touching ids. The result location will be displayed at the end.
 