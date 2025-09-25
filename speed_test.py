import torch
from data import EventSinDataset
import torch_scatter
import tqdm

from models import ModelTransformer, ModelTransformerJagged


def to_nested(batch, x):
    x = [x[batch == b] for b in batch.unique()]
    x = torch.nested.nested_tensor(x, dtype=torch.float, device=batch.device, layout=torch.jagged)
    return x

def positional_embedding(c, n):
    return c[:,None] / torch.arange(n, device=c.device)[None, :]


if __name__ == '__main__':
    dataset = EventSinDataset(num_samples=1000000, time_window_s=1, threshold=0.1,
                                    split="train", sample_rate_hz=100, num_events=100,
                                    set_middle_to_zero=False,
                                    fixed_events=False,
                                    aligned_gt=False,
                                    perturb_window_by_rate=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=dataset.collate)
    model = ModelTransformer().cuda()
    model_jagged = ModelTransformerJagged().cuda()

    lin_in = torch.nn.Linear(1, 128).cuda()

    for (batch_size, max_num_events, counter, batch, timestamps, x, displacement) in loader:

        x = x[:,None].cuda()
        batch = batch.cuda()
        timestamps = timestamps.cuda()
        displacement = displacement.cuda()
        counter = counter.cuda()

        x_nested = to_nested(batch, x)
        c_nested = to_nested(batch, positional_embedding(counter, 128//2))

        for i in tqdm.tqdm(range(10000)):
            output = model_jagged(c_nested, x_nested)
        for i in tqdm.tqdm(range(10000)):
            output = model(batch_size, max_num_events, counter, batch, timestamps, x[:,0])




