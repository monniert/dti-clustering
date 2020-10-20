set -e
CURRENTDATE=`date +"%m%d"`
run_tag="${CURRENTDATE}_$tag"
dataset="$(sed -n 2p configs/$config | cut -f2 -d ':' | cut -c2-)"
CUDA_VISIBLE_DEVICES=$cuda python src/trainer.py --tag $run_tag --config $config
