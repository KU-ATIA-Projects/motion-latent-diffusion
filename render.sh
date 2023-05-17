#!/bin/bash
npy_folder="/home/pjr726/motion-latent-diffusion/results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/random_test"
save_folder = "/home/pjr726/motion-latent-diffusion/example"
cd "$npy_folder"
echo "The directory of npy files are in: $npy_folder"

# remove all the txt files in the directory
rm *.txt
echo "All the txt files are removed"

cd /home/pjr726/motion-latent-diffusion/
python -m fit --dir $npy_folder --save_folder $save_folder
echo "The SMPL meshes are created"

blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=$npy_folder --mode=video --joint_type=HumanML3D
echo "The videos are rendered"

# Save output to a text file
./render.sh > output.txt