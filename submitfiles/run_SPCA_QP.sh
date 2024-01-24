for K in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
for M in 0 1 2
do
    sed -i '$ d' submitfiles/submit_SPCA_QP.sh
    echo "python3 experiments/run_SPCA_QP.py $M $K 0" >> submitfiles/submit_SPCA_QP.sh
    bsub < submitfiles/submit_SPCA_QP.sh
done
done