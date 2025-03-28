InstantSplat Setup Guide
System Requirements
NVIDIA GPU with CUDA support (recommended 12GB+ VRAM)

Conda package manager

Linux (recommended) or Windows with WSL2

Installation
1. Clone Repository and Download Models
bash
复制
git clone https://github.com/your-repo/Enhanc3r.git
cd Enhanc3r

# Download pre-trained model
mkdir -p mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    -P mast3r/checkpoints/
2. Create Conda Environment
bash
复制
conda create -n enhanc3r python=3.10.13 cmake=3.14.0 -y
conda activate enhanc3r
3. Install Dependencies
bash
复制
# Install PyTorch with CUDA 12.1 (adjust version if needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Python dependencies
pip install -r requirements.txt

# Install custom CUDA extensions
pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization/
pip install submodules/fused-ssim/
4. (Optional) Compile RoPE CUDA Kernels
bash
复制
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
Stable Video Diffusion Setup
1. Download Model Weights
Download model weights from Hugging Face

Place them in model/ directory

2. Run Video Processing
bash
复制
python diffusion.py \
    --input_folder output_eval_XL/sora/Santorini/3_views/interp/ours_1000/interp_3_view.mp4 \
    --output_folder output_eval_XL/sora/Santorini/3_views/interp/ours_1000/ \
    --fps 30 \
    --batch_size 1 \
    --mode static
Troubleshooting
Common Issues
CUDA Errors:

Verify CUDA version compatibility: nvcc --version and torch.version.cuda

Reinstall PyTorch matching your CUDA version

Missing Dependencies:

bash
复制
# For Ubuntu/Debian
sudo apt install build-essential cmake
Permission Errors:

bash     
复制
chmod +x scripts/*.sh
Notes
All paths are relative to the project root directory

For optimal performance, use batch_size=1 for high-resolution videos

Monitor GPU memory usage with nvidia-smi

For additional support, please refer to the project documentation or open an issue on GitHub.

Key improvements made:

Added proper Markdown formatting with headers and code blocks

Organized instructions into logical sections

Added troubleshooting section

Included system requirements

Made paths and commands clearer

Added notes about relative paths and performance

Fixed code formatting and line continuations

Added proper spacing for readability

The guide now provides a more professional and complete setup experience while maintaining all the original technical content.