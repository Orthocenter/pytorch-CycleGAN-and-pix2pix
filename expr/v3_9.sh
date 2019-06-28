python train.py \
    --seed=666 \
    --gpu_ids=1 \
    --dataroot=/home/gomezp/rss-data/v3_9 \
    --name=rss_v3_9.c \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=64 \
    --gan_mode=square \
    --netG=unet_64 \
    --display_env=v3_9.c \
    --niter=50 \
    --niter_decay=150 \
    --lr=0.0001 \
    --lambda_enc=0.0 \
    --lambda_syn=0.0 \
    --verbose