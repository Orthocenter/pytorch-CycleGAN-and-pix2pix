python test.py \
    --seed=666 \
    --gpu_ids=3 \
    --dataroot=/home/gomezp/input_synthetic_train_gamma_5.0 \
    --name=rss_v3_2 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=64 \
    --netG=unet_64 \
    --verbose
