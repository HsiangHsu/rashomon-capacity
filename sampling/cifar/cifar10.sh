#!/bin/bash
#SBATCH -p general 	  # Partition to submit to
#SBATCH -n 10 		      # Number of cores
#SBATCH -N 1 		      # Ensure that all cores are on one machine
#SBATCH -t 600    # Runtime in D-HH:MM
#SBATCH --mem=24000   # memory (per node)
#SBATCH -o output.out	# File to which STDOUT will be written
#SBATCH -e errors.err	# File to which STDERR will be written

python3 sample-all.py --dataset 'cifar10' --model 'vgg'  --nrepeat 2 --nepoch 3 --nmodel 50