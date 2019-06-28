python test-rss-v3.py \
    --seed=666 \
    --gpu_ids=3 \
    --dataroot='' \
    --name=rss_v3_10 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=0 \
    --batch_size=1 \
    --netG=unet_64 \
    --verbose
