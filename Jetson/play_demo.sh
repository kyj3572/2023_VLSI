#!/bin/bash
while true; do
    python3 ./trt_video_demo.py --video ms03_vid.mp4
    python3 ./trt_video_demo.py --video Nature_lr.mp4 --model x4_180_320.trt
    python3 ./trt_video_demo.py --video anime_x4_320_180.mp4 --model x4_180_320.trt
done