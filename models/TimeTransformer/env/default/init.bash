[[ ! $PWD == */models/TimeTransformer ]] && echo "Error: Please source this from models/TimeTransformer with the command 'source env/default/init'" && return

git clone https://github.com/Lysarthas/Time-Transformer.git Time_Transformer
(cd Time_Transformer && git checkout ff8faa27353e721702aaa51136192f071e5a5e19)

conda env update -f env/default/environment.yml --prune
conda activate time-transformer
