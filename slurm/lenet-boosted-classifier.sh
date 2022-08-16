#!/bin/bash
#SBATCH --job-name=suepml4
#SBATCH --output=/home/ap6964/suepml/logs/job4.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --mail-user=ap6964@princeton.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

module load anaconda3/2021.11
conda activate solaris

python3 /home/ap6964/suepml/train-classifier.py LeNet5-Boosted-Classifier -c /home/ap6964/suepml/configs/lenet-boosted-classifier.yml
