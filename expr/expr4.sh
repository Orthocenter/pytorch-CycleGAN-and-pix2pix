python train.py \
    --gpu_ids=0 \
    --dataroot=/mnt/data/yxchen/rss-datasets/expr2/train \
    --name=rss_expr4 \
    --model=rssmap2rssmap \
    --input_nc=1 \
    --output_nc=1 \
    --norm=instance \
    --dataset_mode=rss \
    --num_threads=4 \
    --batch_size=32 \
    --gan_mode=vanilla \
    --netG=unet_64 \
    --lambda_L1=0 \
    --display_env=expr4 \
    # --verbose
