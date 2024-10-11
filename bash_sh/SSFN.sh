export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --With_augmentation True \
  --batch_size 32 \
  --mode multimodal \
  --model SSFN \
  --dropout 0 \
  --init_channels [9, 9] \
  --graph_kq_dim [4, 4] \
  --graph_attention_head [2, 2] \
  --fusion_kq_dim [1, 2, 2] \
  --fusion_attention_head [1, 1, 1] \
  --growth_rate 4 \
  --learning_rate 0.0003 \
  --epochs 200 \

