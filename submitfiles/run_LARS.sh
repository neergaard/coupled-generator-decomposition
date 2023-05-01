for O in 0 1 2 3 4
do
for K in 5
do
for M in 0 1 2
do

    sed -i '$ d' submit_LARS.sh
    echo "python3 experiments/run_SPCA_QP.py $M $K $O 0" >> submit_LARS.sh
    bsub < submit_LARS.sh

done
done
done