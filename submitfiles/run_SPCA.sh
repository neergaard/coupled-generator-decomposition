for K in 5
do
for M in 0 1 2
do
for O in 0 1 2 3 4

    sed -i '$ d' submit_SPCA.sh
    echo "python3 experiments/run_SPCA.py $M $K $O" >> submit_SPCA.sh
    bsub < submit_SPCA.sh

done
done
done