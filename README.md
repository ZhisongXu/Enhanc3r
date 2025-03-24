
InstantSplat Setup Guide

Get Started

This guide provides step-by-step instructions to install and set up InstantSplat along with its dependencies.

Installation

1. Clone InstantSplat and Download Pre-trained Model


cd Enhanc3r
mkdir -p mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P mast3r/checkpoints/

2. Create and Activate the Conda Environment

Use Conda to create a new environment and install dependencies.

conda create -n enhanc3r python=3.10.13 cmake=3.14.0 -y
conda activate enhanc3r

3. Install PyTorch and Dependencies

Make sure to install the correct CUDA version for your system.

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # Adjust CUDA version if needed
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim

4. (Optional) Compile CUDA Kernels for RoPE (DUST3R)

DUST3R relies on RoPE positional embeddings, and compiling CUDA kernels improves performance.

cd croco/models/curope/
python setup.py build_ext --inplace

Stable Video Diffusion Setup

1. Download Stable Video Diffusion Model

Download the weights from Hugging Face and place them in the model folder:

mkdir -p model
# Download manually from: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt

2. Run Stable Video Diffusion

Modify the model path in diffusion and run the following command:

python diffusion.py \
    --input_folder output_eval_XL/sora/Santorini/3_views/interp/ours_1000/interp_3_view.mp4 \
    --output_folder output_eval_XL/sora/Santorini/3_views/interp/ours_1000/ \
    --fps 30 \
    --batch_size 1 \
    --mode static

This will process the input video and generate the enhanced output.

Notes

Ensure all dependencies are properly installed before running the scripts.

Modify paths as needed based on your project structure.

For CUDA-related issues, check PyTorch and CUDA compatibility.

Enjoy using InstantSplat and Stable Video Diffusion!