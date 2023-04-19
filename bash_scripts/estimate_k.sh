nvidia-smi
hostname
export CUDA_VISIBLE_DEVICES=6,7
# Get unique log file
SAVE_DIR=/raid/zhaochuan/master/NCL/code/OWSOL/log/estimate_k/
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m methods.estimate_k \
        --max_classes 5000 \
        --batch_size 64 \
        --dataset_name 'inatloc' \
        --search_mode  'brent'\
> ${SAVE_DIR}logfile_${EXP_NUM}.log