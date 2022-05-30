#!/bin/bash
#SBATCH -p general 	  # Partition to submit to
#SBATCH -n 10 		      # Number of cores
#SBATCH -N 1 		      # Ensure that all cores are on one machine
#SBATCH -t 600    # Runtime in D-HH:MM
#SBATCH --mem=24000   # memory (per node)
#SBATCH -o output.out	# File to which STDOUT will be written
#SBATCH -e errors.err	# File to which STDERR will be written

python3 perturb-all.py --dataset 'cifar10' --model 'vgg' --nepoch 3 --nrepeat 2 --loss_tolerance 0.01 --test_split 0.01
python3 perturb-all.py --dataset 'cifar10' --model 'vgg' --nepoch 3 --nrepeat 2 --loss_tolerance 0.02 --test_split 0.01
python3 perturb-all.py --dataset 'cifar10' --model 'vgg' --nepoch 3 --nrepeat 2 --loss_tolerance 0.05 --test_split 0.01
python3 perturb-all.py --dataset 'cifar10' --model 'vgg' --nepoch 3 --nrepeat 2 --loss_tolerance 0.1 --test_split 0.01