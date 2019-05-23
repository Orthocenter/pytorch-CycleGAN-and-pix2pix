python train.py \
    --seed=666 \
    --gpu_ids=3 \
    --dataroot=/home/gomezp/rss-data/1.1\
    --name=rss_v3_1.1 \
    --model=rss2rssmerged\
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=64 \
    --gan_mode=square \
    --netG=v3_unet_64 \
    --lambda_L1=0 \
    --lambda_T=100 \
    --lr=0.00004 \
    --display_env=v3_1.1 \
    --niter=50 \
    --niter_decay=150 \
    --verbose
