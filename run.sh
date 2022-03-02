#!/bin/bash

rm -rf annotations* bbox* rectangles.pom calibrations

cp -R ../seq1/calibrations calibrations
scp -r /scratch/jeetv/scene5/config1/seq5/Image_subsets/cam_1.mp4 Image_subsets/cam_5.mp4
scp -r /scratch/jeetv/scene5/config1/seq5/Image_subsets/cam_6.mp4 Image_subsets/cam_6.mp4
scp -r /scratch/jeetv/scene5/config1/seq5/Image_subsets/cam_3.mp4 Image_subsets/cam_7.mp4
scp -r /scratch/jeetv/scene5/config1/seq5/Image_subsets/cam_2.mp4 Image_subsets/cam_8.mp4

scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera1_3d.txt matchings/Camera5_3d.txt
scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera6_3d.txt matchings/Camera6_3d.txt
scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera3_3d.txt matchings/Camera7_3d.txt
scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera2_3d.txt matchings/Camera8_3d.txt

scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera1.txt matchings/Camera5.txt
scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera6.txt matchings/Camera6.txt
scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera3.txt matchings/Camera7.txt
scp -r /scratch/jeetv/scene5/config1/seq5/matchings/Camera2.txt matchings/Camera8.txt
