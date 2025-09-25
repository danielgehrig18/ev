from pathlib import Path
import tqdm
from ev.dataset import Dataset


if __name__ == '__main__':
    dataset = Dataset(root=Path("/tmp/train_spline/"), split="train")
    for i in tqdm.tqdm(range(len(dataset))):
        data = dataset[i]
