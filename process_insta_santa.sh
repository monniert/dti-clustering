set -e
cur_dir=$(pwd)
mkdir -p datasets/instagram
cd datasets/instagram && instagram-scraper --tag santaphoto --maximum 10000 --media-types image
mkdir raw && mv santaphoto/* raw && mv raw santaphoto/
cd $cur_dir
python src/img_resizer.py -i datasets/instagram/santaphoto/raw -o datasets/instagram/santaphoto/train -s 128
