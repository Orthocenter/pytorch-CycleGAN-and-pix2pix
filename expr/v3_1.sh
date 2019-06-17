python train.py \
    --seed=666 \
    --gpu_ids=3 \
    --dataroot=/home/gomezp/input_synthetic_train_gamma_5.0 \
    --name=rss_v3_1 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=64 \
    --gan_mode=square \
    --netG=unet_64 \
    --display_env=v3_1 \
    --niter=100 \
    --niter_decay=100 \
    --lr=0.0001 \
    --verbose
