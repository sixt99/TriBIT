#!/bin/bash
#SBATCH --account bsc03
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time 01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1



srun --gres=gpu:4 src/multi_gpu/target/release/rs data/multi_gpu/gsh-2015-host/gsh-2015-host 64; exit






# EXTRA
srun --gres=gpu:4 target/release/rs ../graph/it-2004 34; exit
srun --gres=gpu:4 target/release/rs ../graph/network 256; exit
srun --gres=gpu:4 target/release/rs ../graph/clueweb12 180; exit

# SCALABILITY
srun --gres=gpu:1 target/release/rs ../graph/eu-2015 128; exit
srun --gres=gpu:1 target/release/rs ../graph/webgraph 128; exit
srun --gres=gpu:1 target/release/rs ../graph/uk-2014 64; exit
srun --gres=gpu:1 target/release/rs ../graph/gsh-2015 64; exit
srun --gres=gpu:1 target/release/rs ../graph/uk-union-2006-06-2007-05-underlying 128; exit
srun --gres=gpu:1 target/release/rs ../graph/uk-2007-02 64; exit
srun --gres=gpu:1 target/release/rs ../graph/uk-2007-01-hc 64; exit
srun --gres=gpu:1 target/release/rs ../graph/uk-2006-09 64; exit
srun --gres=gpu:1 target/release/rs ../graph/gsh-2015-host 128; exit
srun --gres=gpu:1 target/release/rs ../graph/twitter-2010 128; exit

# SCALABILITY COMPARISON
srun --gres=gpu:$1 target/release/rs ../graph/scalability/arabic-2005.graph 64; exit
srun --gres=gpu:$1 target/release/rs ../graph/scalability/indochina-2004.graph 64; exit
srun --gres=gpu:$1 target/release/rs ../graph/scalability/it-2004.graph 64; exit
srun --gres=gpu:$1 target/release/rs ../graph/scalability/sk-2005.graph 64; exit
srun --gres=gpu:$1 target/release/rs ../graph/scalability/uk-2002.graph 64; exit
srun --gres=gpu:$1 target/release/rs ../graph/scalability/uk-2005.graph 64; exit

