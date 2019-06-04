python train.py \
    --seed=666 \
    --gpu_ids=3 \
    --dataroot=/mnt/data/yxchen/rss-datasets/expr15.2/train \
    --name=rss_expr16.3 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=256 \
    --gan_mode=square \
    --netG=unet_64 \
    --lambda_L1=0 \
    --lambda_T=100 \
    --lambda_SYM=10 \
    --blocked_size=0 \
    --lr=0.00002 \
    --display_env=expr16.3 \
    --niter=50 \
    --niter_decay=150 \
    --raytracing_A=True \
    --verbose