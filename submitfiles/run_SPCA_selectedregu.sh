for O in 0 1 2 3 4
do
for M1 in 0 1
do
for M2 in 0 1
do

    sed -i '$ d' submit_SPCA_selectedregu.sh
    echo "python3 experiments/run_SPCA_QP_selectedregu.py $M1 $M2 $O" >> submit_SPCA_selectedregu.sh
    bsub < submit_SPCA_selectedregu.sh

done
done
done