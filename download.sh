#!/bin/bash

rsync -aP ada:/share3/jeet.vora/GTA_scene1/GTA_scene1_A/Image_subsets /scratch/jeetv/GTA_scene1_A
rsync -aP ada:/share3/jeet.vora/GTA_scene1/GTA_scene1_B/Image_subsets /scratch/jeetv/GTA_scene1_B
rsync -aP ada:/share3/jeet.vora/GTA_scene2/GTA_scene2_A/Image_subsets /scratch/jeetv/GTA_scene2_A
rsync -aP ada:/share3/jeet.vora/GTA_scene2/GTA_scene2_B/Image_subsets /scratch/jeetv/GTA_scene2_B

rsync -aP ada:/share3/jeet.vora/GTA_Sample/gta_sampleA/ /scratch/jeetv/GTA_scene1_A/
rsync -aP ada:/share3/jeet.vora/GTA_Sample/gta_sampleB/ /scratch/jeetv/GTA_scene1_B/
rsync -aP ada:/share3/jeet.vora/GTA_Sample/gta_sampleC/ /scratch/jeetv/GTA_scene2_A/
rsync -aP ada:/share3/jeet.vora/GTA_Sample/gta_sampleD/ /scratch/jeetv/GTA_scene2_B/
rsync -aP ada:/share3/jeet.vora/GTA_Sample/GTA_SampleE/ /scratch/jeetv/GTA_scene3_A
echo "File transfered"
cd /scratch/jeetv/
echo "MVDet unzipped.."
mv GTA_scene* GMVD

