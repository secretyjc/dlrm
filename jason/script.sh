
git config --local --add core.sshCommand 'ssh -i ~/.ssh/id2'

python data_utils.py --raw-data-file /disk/kaggle/train.txt --processed-data-file /disk/kaggle/kaggleAdDisplayChallenge_processed.npz --memory-map
python data_utils.py --raw-data-file /mntData2/mlsys/dlrm/criteo/day --processed-data-file /mntData2/mlsys/dlrm/criteo/processed/terabyte_processed.npz 

python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --save-model model.ka --dataset-multiprocessing --print-time $dlrm_extra_option
