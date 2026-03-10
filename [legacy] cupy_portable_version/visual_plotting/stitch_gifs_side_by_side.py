#!/usr/bin/env python3
"""
stitch_gifs_side_by_side.py
Goal: Stitches two or more GIFs side-by-side into a composite animation for visual comparison.
"""
import sys
import os
from PIL import Image
import imageio


def stitch_gifs_side_by_side(gif_paths, output_path):
    # Load all GIFs as lists of frames
    gifs = [imageio.mimread(gif_path) for gif_path in gif_paths]
    n_frames = min(len(frames) for frames in gifs)
    # Resize all frames to match the first GIF
    base_size = gifs[0][0].size if isinstance(gifs[0][0], Image.Image) else gifs[0][0].shape[::-1]
    stitched_frames = []
    for i in range(n_frames):
        # Convert all frames to PIL Images and resize
        pil_frames = [Image.fromarray(frames[i]) if not isinstance(frames[i], Image.Image) else frames[i] for frames in gifs]
        pil_frames = [frame.resize(base_size) for frame in pil_frames]
        # Concatenate horizontally
        total_width = base_size[0] * len(pil_frames)
        new_img = Image.new('RGB', (total_width, base_size[1]))
        for idx, frame in enumerate(pil_frames):
            new_img.paste(frame, (idx * base_size[0], 0))
        stitched_frames.append(new_img)
    # Save as GIF
    stitched_frames[0].save(output_path, save_all=True, append_images=stitched_frames[1:], duration=100, loop=0)
    print(f"SUCCESS! Stitched GIF saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    gifs_to_stitch = ["basin_evolution.gif", "basin_comparison.gif"]
    output_gif = "stitched_comparison.gif"
    stitch_gifs_side_by_side(gifs_to_stitch, output_gif)
