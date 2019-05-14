python train.py \
    --seed=666 \
    --gpu_ids=3 \
    --dataroot=/mnt/data/yxchen/rss-datasets/expr8/train \
    --name=rss_expr9.7 \
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
    --blocked_size=20 \
    --lr=0.00004 \
    --display_env=expr9.7 \
    --niter=50 \
    --niter_decay=150 \
    --verbose