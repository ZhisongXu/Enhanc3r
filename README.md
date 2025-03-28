
# Enhanc3r Installation Guide

## 1. Clone Repository and Download Models

```bash
git clone https://github.com/your-repo/Enhanc3r.git
cd Enhanc3r

# Download pre-trained model
mkdir -p mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \    
    -P mast3r/checkpoints/
```

## 2. Create Conda Environment

```bash
conda create -n enhanc3r python=3.10.13 cmake=3.14.0 -y
conda activate enhanc3r
```

## 3. Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 (adjust version if needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Python dependencies
pip install -r requirements.txt

# Install custom CUDA extensions
pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization/
pip install submodules/fused-ssim/
```

## 4. (Optional) Compile RoPE CUDA Kernels

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

---

# Stable Video Diffusion Setup

## 1. Download Model Weights
Download model weights from Hugging Face and place them in the `model/` directory.

## 2. Run Video Processing

```bash
python diffusion.py \
    --input_folder output_eval_XL/sora/Santorini/3_views/interp/ours_1000/interp_3_view.mp4 \
    --output_folder output_eval_XL/sora/Santorini/3_views/interp/ours_1000/ \
    --fps 30 \
    --batch_size 1 \
    --mode static
```


