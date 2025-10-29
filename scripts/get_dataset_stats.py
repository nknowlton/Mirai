from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
from onconet.utils.parsing import parse_args
from onconet.utils.get_dataset_stats import get_dataset_stats

"""Compute dataset channel statistics.

Parses dataset arguments and prints the mean and standard deviation of the
images in the dataset. This is a thin wrapper around ``onconet.utils.get_dataset_stats``.

Usage
-----
```bash
python scripts/get_dataset_stats.py --dataset <name> --img_dir <dir> --metadata_dir <dir>
```
"""


if __name__ == "__main__":
    # REQUIRED ARGS:
    # - dataset
    # - img_dir
    # - metadata_dir
    # OPTIONAL ARGS:
    # - cuda
    # - max_batches_per_epoch
    # - train_years
    # - dev_years
    # - test_years

    args = parse_args()
    means, stds = get_dataset_stats(args)
    for channel, (mean, std) in enumerate(zip(means, stds)):
        print("Channel[{}] Img mean:{}, Img std:{}".format(channel, means, std))
