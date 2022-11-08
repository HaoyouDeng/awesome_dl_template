# for single gpu
GPU_ID=0
python main.py ./configs/basic.ymp --gpus $GPU_ID

# change config
GPU_ID=0
python main.py ./configs/basic.ymp --gpus $GPU_ID \
name="experiment name" \
loss.weight.l1=2 \
train.num_epoch=100

# for multi gpus
CUDA_VISIBLE_DEVICE=1,2
GPU_COUNT=2
torchrun --nproc_pre_node=$GPU_COUNT main.py ./configs/basic.ymp --gpus 'all' \
train.dataloader.shuffle=False