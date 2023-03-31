for M in 0
do
for K in 2
do

    sed -i '$ d' submit_SPCA.sh
    echo "python3 run_SPCA.py $M $K" >> submit_SPCA.sh
    bsub < submit_SPCA.sh

done
done