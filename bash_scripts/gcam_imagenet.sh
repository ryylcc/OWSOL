hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=3

python -m methods.gcam \
        --batch_size 128 \
        --dataset_name 'ImageNet' \
        --model_path './save/ImageNet/lr0.0005_scl0.5_mcl1.0_mc5_e10/checkpoint_0009.pth.tar' \
        --partitions 'all'  'known'  'nov_s'  'nov_d'