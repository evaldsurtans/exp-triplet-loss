#!/bin/sh -v
#PBS -e /mnt/home/evaldsu/data_raw/vggface2/err.txt
#PBS -o /mnt/home/evaldsu/data_raw/vggface2/out.txt
#PBS -q batch
#PBS -p 1000
#PBS -l nodes=1:ppn=32
#PBS -l mem=100gb
#PBS -l walltime=96:00:00

module load conda
eval "$(conda shell.bash hook)"
source activate conda_env
mkdir /scratch/evalds
mkdir /scratch/evalds/tmp
mkdir /scratch/evalds/data
export TMPDIR=/scratch/evalds/tmp
export TEMP=/scratch/evalds/tmp
export SDL_AUDIODRIVER=waveout
export SDL_VIDEODRIVER=x11

cd ~/Documents/fassion_minst/

python ./process_data_simpsons.py \
-path_input_train /mnt/home/evaldsu/data_raw/simpsons/simpsons_dataset \
-path_input_test /mnt/home/evaldsu/data_raw/simpsons/kaggle_simpson_testset/kaggle_simpson_testset \
-path_output /mnt/home/evaldsu/data_raw/simpsons_128_c



