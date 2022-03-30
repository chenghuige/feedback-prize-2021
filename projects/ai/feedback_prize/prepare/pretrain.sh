python run_mlm.py \
    --model_name_or_path $1 \
    --train_file ../input/feedback-prize-2021/train_corpus.txt \
    --validation_file ../input/feedback-prize-2021/train_corpus.txt \
    --do_train \
    --do_eval \
    --output_dir ../input/feedback-prize-2021/pretrain/$1
