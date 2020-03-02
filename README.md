# CS224N Project

To run, execute `run_squad.py` with desired arguments.

## Running on colab

Checkpoints are saved to Google Drive. Note that the code will periodically delete old checkpoints, but you need to manually delete them from the trash in Google Drive to avoid consuming quota.

```python
!pip install transformers
%load_ext autoreload
%autoreload 2
import run_squad
argv = ('--model_type albert --model_name_or_path albert-base-v2 --do_train '+
        '--do_eval --do_lower_case --version_2_with_negative --train_file data/train-v2.0.json '+
        '--predict_file data/dev-v2.0.json --per_gpu_train_batch_size 12 '+
        '--learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 '+
        '--doc_stride 128').split() + [
        # Note the space in the string, can't put in split()
        '--output_dir', '/content/drive/My Drive/cs224n/albert_uncased_output'
        ]
run_squad.main(argv)
```
