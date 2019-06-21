COM="sbatch --gres=gpu:1 --time=7200 --mem=14000 ./jperson.sh"
echo $COM
eval $COM
