module purge 

python -m venv env/sc_uni_leipzig/venv
source env/sc_uni_leipzig/venv/bin/activate
pip install -r requirements.txt

module load torchvision
module load TensorFlow
