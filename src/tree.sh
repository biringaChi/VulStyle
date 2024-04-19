# Parse c/c++ functions, generates and dumps ASTs to a text file
# Used in pre-training and fine-tuning datasets

path=`find $PWD/pretrain/ -name "*.c"`

for file in $path; do
   clang -cc1 -ast-dump $file >> "$file.txt"
done