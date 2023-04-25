for K in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
for M in 0 1 2
do

    sed -i '$ d' submit_SPCA.sh
    echo "python3 experiments/run_SPCA_noregu.py $M $K" >> submit_SPCA.sh
    bsub < submit_SPCA.sh

done
done