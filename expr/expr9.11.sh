python train.py \
    --seed=666 \
    --gpu_ids=2 \
    --dataroot=/mnt/data/yxchen/rss-datasets/expr8/train \
    --name=rss_expr9.11 \
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
    --lambda_T=100 \
    --blocked_size=22 \
    --lr=0.00004 \
    --display_env=expr9.11 \
    --niter=50 \
    --niter_decay=150 \
    --verbose
