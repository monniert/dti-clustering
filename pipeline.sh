set -e
CURRENTDATE=`date +"%m%d"`
run_tag="${CURRENTDATE}_$tag"
CUDA_VISIBLE_DEVICES=$cuda python src/trainer.py --tag $run_tag --config $config
