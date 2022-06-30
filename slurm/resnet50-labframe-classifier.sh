#!/bin/bash
#SBATCH --job-name=suepml3
#SBATCH --output=/home/ap6964/suepml/logs/job3.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:4
#SBATCH --time=28:00:00
#SBATCH --mail-user=ap6964@princeton.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

module load anaconda3/2021.11
conda activate solaris

python3 /home/ap6964/suepml/train-classifier.py ResNet50-LabFrame-Classifier -c /home/ap6964/suepml/configs/resnet50-labframe-classifier.yml
