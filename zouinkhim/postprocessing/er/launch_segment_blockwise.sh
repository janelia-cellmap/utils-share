rm nohup.out
rm -rf daisy_logs
rm -rf task.*
# ulimit -n 6000
# nohup python segment_blockwise.py &

bsub -P cellmap -J segment_ecs_master -n 20 -o task.out -e task.err python segment_blockwise.py
