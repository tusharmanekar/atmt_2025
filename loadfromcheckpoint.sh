#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=6:0:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=resumed_assignment3.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Just to be safe, ensure dirs still exist
mkdir -p assignment3/cz-en/data/prepared
mkdir -p assignment3/cz-en/logs
mkdir -p assignment3/cz-en/checkpoints

python train.py \
    --cuda \
    --data assignment3/cz-en/data/prepared/ \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 1 \
    --log-file assignment3/cz-en/logs/train_resume.log \
    --save-dir assignment3/cz-en/checkpoints/ \
    --restore-file checkpoint_last.pt \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path assignment3/cz-en/checkpoints/checkpoint_best.pt \
    --batch-size 1 \
    --max-len 300 \
    --output assignment3/cz-en/output.txt \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en 