# In Context Function Estimation
Install project with
````
pip install .
````
 ## Sample random functions
Functions are sampled from a multivariate normal distribution and a RBF kernel.
The length scale value is sampled from a beta distribution, with its parameters alpha and beta required as 
command line arguments. The number of context points is sampled from a uniform distribution, with min and max values 
controllable as command line arguments.
```` Python
python -m icfelab.sample -f data/file.xz -n 10 --alpha 2 --beta 5 --min 5 --max 50
````

 ## Train and evaluate transformer based estimator
Training and evaluation are combined in one script. Model configurations and hyperparameters are loaded from a yml config file.
For saved models, the corresponding config file is saved alongside and loaded automaticly while evaluating.
Training can be done in the default mode, directly predicting function values, as well as the gaussian mode, predicting mean and log variance.

Train:
```` Python
python -m icfelab.train -e 200 -n train_run -d data/data.xz -w 4 --config_path config/config.yml --gaussian
````

Evaluation:
```` Python
python -m icfelab.train -n test -d data/data.xz -cp config/config.yml --gaussian -w 4 --eval models/model_A --full-eval
````

In both modes, a hardcoded amount of functions is plotted into a folder with the same name, as the 
provided name (-n) argument.
