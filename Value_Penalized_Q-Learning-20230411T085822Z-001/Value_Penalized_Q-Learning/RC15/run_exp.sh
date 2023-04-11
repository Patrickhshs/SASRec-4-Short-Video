#!bin/bash
# Reproduce the paper results.

conda activate tf113

mkdir you_own_log_dir

for (i=0;i<5;i++)
do
python GRU_AC_VPQ.py --method rem --coef 20 --out gru_ac_vpq_$i --gpu 0
python Caser_AC_VPQ.py --method rem --coef 20 --out caser_ac_vpq_$i --gpu 0
python NextItNet_AC_VPQ.py --method rem --coef 20 --out next_ac_vpq_$i --gpu 0
python SASRec_AC_VPQ.py --method rem --coef 20 --out sasrec_ac_vpq_$i --gpu 0
done