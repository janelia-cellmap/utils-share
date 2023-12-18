rm nohup.out
rm -rf daisy_logs
ulimit -n 6000
nohup python segment_blockwise.py &