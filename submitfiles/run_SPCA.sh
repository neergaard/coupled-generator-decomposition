for M in 0 1 2
do
for K in 2 3 4 5 6 7 8 9 10
do

    sed -i '$ d' submit_SPCA.sh
    echo "python3 experiments/run_SPCA.py $M $K" >> submit_SPCA.sh
    bsub < submit_SPCA.sh

done
done
