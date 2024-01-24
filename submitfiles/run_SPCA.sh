for K in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
for M in 0 1 2
do
for I in 0 1
do
    sed -i '$ d' submitfiles/submit_SPCA.sh
    echo "python3 experiments/run_SPCA.py $M $K $I" >> submitfiles/submit_SPCA.sh
    bsub < submitfiles/submit_SPCA.sh

done
done
done