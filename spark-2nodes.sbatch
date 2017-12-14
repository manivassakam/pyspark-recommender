#!/bin/bash

#SBATCH --account=msca32017
#SBATCH --job-name=mani_recommender_project
#SBATCH --output=log_%j.out.txt
#SBATCH --error=log_%j.err.txt
#SBATCH --time=10:30:00
# Exclusive mode is recommended for all spark jobs
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --partition=broadwl 


module load spark/2.2 Anaconda3

# This command starts the spark workers on the allocated nodes
start-spark-slurm.sh

# This syntax tells spark to use all cpu cores on the node.
export MASTER=spark://$HOSTNAME:7077
export PYTHONHASHSEED=0
# This is a python example. 
# For production jobs, you'll probably want to have a python module loaded.
# This will use the system python if you don't have a python module loaded.
spark-submit --master $MASTER --driver-memory 2g --executor-memory 6g RecommenderProject_FittingALS_2-Copy1.py
