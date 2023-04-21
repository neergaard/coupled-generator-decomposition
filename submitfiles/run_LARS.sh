for K in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do

    sed -i '$ d' submit_LARS.sh
    echo "python3 experiments/run_SPCA_lars.py $K" >> submit_LARS.sh
    bsub < submit_LARS.sh

done

