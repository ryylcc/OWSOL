hostname
nvidia-smi
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Get unique log file,
SAVE_DIR='/raid/xjheng/zc/NCL/code/OWSOL/log/ImageNet/'
mkdir -p  ${SAVE_DIR}
LR='0.003'
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM


python -m methods.contrastive_co_training \
            --dataset_name 'ImageNet' \
            -b 256 \
            --lr ${LR} \
            --epochs 10 \
            --dist_url 'tcp://localhost:10001' \
            --multiprocessing_distributed --world_size 1 --rank 0 \
            --num_cluster 50000 \
            --mcl_k 16384 \
            --num_multi_centroids 5 \
            --scl_weight 0.5 \
            --mcl_weight 1.0 \
> ${SAVE_DIR}logfile_${EXP_NUM}_${LR}.log


