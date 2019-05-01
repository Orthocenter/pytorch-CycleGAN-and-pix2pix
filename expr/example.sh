python train.py \
    --seed=666 \
    --gpu_ids=0 \
    --dataroot=/mnt/data/yxchen/rss-datasets/expr8/train \
    --name=rss_expr8.1 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=64 \
    --gan_mode=square \
    --netG=unet_64 \
    --lambda_L1=0 \
    --lambda_T=1 \
    --display_env=expr8.1 \
    --n_iter=100 \
    --n_iter_decay=100 \
    --max_B_size=20 \
    --verbose
