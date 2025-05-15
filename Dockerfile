from docker.io/python:3.7

# TSGBench dependencies
run pip install numpy
run pip install matplotlib
run pip install scipy
run pip install dtaidistance
run pip install tensorflow
run pip install scikit-learn
run pip install torch
run pip install statsmodels
run pip install tslearn
run pip install seaborn
run pip install mgzip
run pip install pyyaml
run pip install pandas

# Development utilities
run pip install jupyterlab