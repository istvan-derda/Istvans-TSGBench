[[ ! $PWD == */models/TimeGAN ]] && echo "Error: Please source this from models/TimeGAN with the command 'source env/default/init" && return

git clone https://github.com/jsyoon0823/TimeGAN.git Time_GAN
(cd Time_GAN && git checkout 8f6181cb9b9d2fa0c930cd902411d9ac8a308e07)
sed -i '22s/from utils/from .utils/' Time_GAN/timegan.py # patch to make utils.py import work when timegan.py is imported from parent directory

conda env update -f env/default/environment.yml --prune
conda activate time-gan
