python train.py \
    --gpu_ids=2 \
    --dataroot=/mnt/data/yxchen/rss-datasets/expr2.1/train \
    --name=rss_expr4.8 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=batch \
    --dataset_mode=rss \
    --num_threads=4 \
    --batch_size=64 \
    --gan_mode=vanilla \
    --netG=unet_64 \
    --lambda_L1=0 \
    --lambda_T=100 \
    --display_env=expr4.8 \
    --verbose