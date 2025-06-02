#!/bin/sh
if [ ! -d Time_Transformer ]
then
    git clone https://github.com/Lysarthas/Time-Transformer.git Time_Transformer
fi
(cd Time_Transformer && git checkout ff8faa27353e721702aaa51136192f071e5a5e19)
conda env remove -n time-transformer -y
conda env create -f environment.yml
conda activate time-transformer