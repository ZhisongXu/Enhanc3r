import os
import argparse
import numpy as np
import imageio
import torch
import cv2
from PIL import Image
from typing import List, Union
from svd.svd_pipeline import StableVideoDiffusionPipeline
from svd.scheduler import EulerDiscreteScheduler
from render import images_to_video
import logging

def video_to_numpy(video_path, device='cuda', target_height=224, target_width=224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (target_width, target_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_tensor = torch.tensor(frame_normalized).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
        frames.append(frame_tensor)

    cap.release()

    video_tensor = torch.cat(frames, dim=0)  # [num_frames, 3, H, W]
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: [1, num_frames, 3, H, W]
    return video_tensor


def export_to_video(
    video_frames: Union[List[np.ndarray], List[Image.Image]], output_path: str = None , fps: int = 10, H=None, W=None
) -> str:
    os.makedirs(f'{output_path}/images', exist_ok=True)
    output_video_path = os.path.join(output_path, f'video.mp4')

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = video_frames[0].shape[0:2] if H is None and W is None else (H, W)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        if H is not None and W is not None:
            img = cv2.resize(img, (W, H))
        cv2.imwrite(f"{output_path}/images/frames_{i:02d}.png", img)
        video_writer.write(img)
    return output_video_path


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def check_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # In GB
    cached_memory = torch.cuda.memory_reserved() / 1024**3  # In GB
    logger.debug(f"Memory allocated: {allocated_memory:.2f} GB")
    logger.debug(f"Memory cached: {cached_memory:.2f} GB")

def process_video_pipeline(input_folder, output_folder, fps=25, batch_size=4, mode='static'):
    try:
        logger.debug(f"Setting up output folder: {output_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        images_folder = os.path.join(output_folder, 'images')
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        logger.debug("Loading Stable Video Diffusion model...")
        check_memory_usage()
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "/home/logan-kyo/models/svd", torch_dtype=torch.float16
        ).to("cuda")
        check_memory_usage()

        logger.debug("Enabling memory optimizations...")
        pipe.enable_attention_slicing(1)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        batch_size = min(2, batch_size)
        logger.debug(f"Using batch size: {batch_size}")

        logger.debug(f"Loading video from: {input_folder}")
        video = video_to_numpy(input_folder, device='cuda')
        num_frames = video.shape[1]  # Dynamically get number of frames (201 in your case)
        H, W = video.shape[3], video.shape[4]  # Adjust for new shape
        logger.debug(f"Video loaded with shape: {video.shape}")

        check_memory_usage()

        logger.debug("Setting random seed generator for reproducibility...")
        generator = torch.manual_seed(42)

        logger.debug("Running Stable Video Diffusion on video...")
        frames = pipe(video,
                      num_frames=num_frames,  # Explicitly pass num_frames
                      decode_chunk_size=8,
                      fps=6,
                      generator=generator,
                      num_inference_steps=25,
                      resample_steps=3,
                      guide_util=15 if mode == 'static' else 16).frames[0]

        logger.debug("Finished generating frames. Exporting frames to images...")
        output_video_path = export_to_video(frames, output_folder, fps)
        logger.debug(f"Processing completed! Output video saved at: {output_video_path}")
        return output_video_path

    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}")
        raise


def main():
    """Main function to handle command-line arguments and execute video processing."""
    # Parse command-line arguments
    print("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Process a video with Stable Video Diffusion.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder where results will be saved')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate of the generated video')
    parser.add_argument('--batch_size', type=int, default=4, help='Processing batch size')
    parser.add_argument('--mode', type=str, choices=['static', 'dynamic'], default='static', help="Processing mode: 'static' or 'dynamic'")
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    # Call the video processing function with the provided arguments
    print("Starting the video processing pipeline...")
    process_video_pipeline(args.input_folder, args.output_folder, args.fps, args.batch_size, args.mode)

if __name__ == '__main__':
    print("Starting script execution...")
    main()