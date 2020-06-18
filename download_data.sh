set -e
# affNIST-test
wget 'https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip' --output-document test.mat.zip
unzip test.mat.zip && rm test.mat.zip
mkdir -p datasets
mv test.mat datasets/affNIST_test.mat

# FRGC
wget 'https://github.com/XifengGuo/JULE-Torch/blob/master/datasets/FRGC/data4torch.h5' --output-document data4torch.h5
mkdir -p datasets/FRGC
mv data4torch.h5 datasets/FRGC
