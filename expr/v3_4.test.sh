python test.py \
    --seed=666 \
    --gpu_ids=1 \
    --dataroot=/home/gomezp/testing_real_emu_train_gamma_5.0 \
    --name=rss_v3_4 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=12 \
    --batch_size=64 \
    --netG=unet_64 \
    --verbose
