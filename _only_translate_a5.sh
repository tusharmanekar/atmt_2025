#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=2:35:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_only_out.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE for different alpha values on a small subset
for alpha in 0.0 0.5 1.0; do
    echo "Running translation with alpha = ${alpha}"
    python translate_a5.py \
        --cuda \
        --input /home/tmanek/atmt_subsets/test_small.cz \
        --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
        --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
        --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
        --output cz-en/output/output_alpha_${alpha}.txt \
        --max-len 300 \
        --beam-size 5 \
        --alpha ${alpha}
done