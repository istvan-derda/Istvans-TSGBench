[[ ! $PWD == */models/Time-Transformer\ AAE ]] && echo "Error: Please source this from models/Time-Transformer AAE with the command 'source env/sc_uni_leipzig/init" && return

git clone https://github.com/Lysarthas/Time-Transformer.git Time_Transformer
(cd Time_Transformer && git checkout ff8faa27353e721702aaa51136192f071e5a5e19)

module purge
conda env update -f env/sc_uni_leipzig/environment.yml --prune
conda activate time-transformer
module load TensorFlow
