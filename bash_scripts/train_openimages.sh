hostname
nvidia-smi
export CUDA_VISIBLE_DEVICES=4,5
# Get unique log file,
SAVE_DIR='/home/zhaochuan/workspace/NCL/code/OWSOL/log/OpenImages/'
mkdir -p  ${SAVE_DIR}
LR='0.001'
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM


python -m methods.contrastive_co_training \
            --dataset_name 'OpenImages' \
            -b 64 \
            --lr 0.001 \
            --epochs 10 \
            --dist_url 'tcp://localhost:10001' \
            --multiprocessing_distributed --world_size 1 --rank 0 \
            --num_cluster 2500 \
            --mcl_k 2048 \
            --num_multi_centroids 5 \
            --scl_weight 0.5 \
            --mcl_weight 1.0 \
> ${SAVE_DIR}logfile_${EXP_NUM}_${LR}.log


