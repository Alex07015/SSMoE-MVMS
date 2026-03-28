import torch
from torch_geometric.data import Batch

def pad_1d_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """
    padding one dimension tokens inputs.

    :param values: A list of 1d tensors.
    :param pad_idx: The padding index.
    :param left_pad: Whether to left pad the tensors. Defaults to False.
    :param pad_to_length: The desired length of the padded tensors. Defaults to None.
    :param pad_to_multiple: The multiple to pad the tensors to. Defaults to 1.

    :return: A padded 1d tensor as a torch.Tensor.

    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def pad_coords(
    values,
    pad_idx,
    dim=3,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """
    padding twozzzzxZXsafaqqqqqqqaaaa dimension tensor coords which the third dimension is 3.

    :param values: A list of 1d tensors.
    :param pad_idx: The value used for padding.
    :param left_pad: Whether to pad on the left side. Defaults to False.
    :param pad_to_length: The desired length of the padded tensor. Defaults to None.
    :param pad_to_multiple: The multiple to pad the tensor to. Defaults to 1.

    :return: A padded 2d coordinate tensor as a torch.Tensor.
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


def pad_2d(
    values,
    pad_idx,
    dim=1,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """
    padding two dimension tensor inputs.

    :param values: A list of 2d tensors.
    :param pad_idx: The padding index.
    :param left_pad: Whether to pad on the left side. Defaults to False.
    :param pad_to_length: The length to pad the tensors to. If None, the maximum length in the list
                         is used. Defaults to None.
    :param pad_to_multiple: The multiple to pad the tensors to. Defaults to 1.

    :return: A padded 2d tensor as a torch.Tensor.
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if dim == 1:
        res = values[0].new(len(values), size, size).fill_(pad_idx)
    else:
        res = values[0].new(len(values), size, size, dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            (
                res[i][size - len(v) :, size - len(v) :]
                if left_pad
                else res[i][: len(v), : len(v)]
            ),
        )
    return res


def Multi_process_batch_collate_fn(samples):
    """
    Custom collate function for batch processing non-MOF data.

    :param samples: A list of sample data.

    :return: A tuple containing a batch dictionary and labels.
    """
    unimol_batch = {}
    data_unimol = []
    data_pyg = []
    for data in samples:
        data_unimol.extend(data['unimol'])
        data_pyg.extend(data['pyg'])

    for k in data_unimol[0].keys():
        if k == 'src_coord':
            v = pad_coords(
                [s[k] for s in data_unimol], pad_idx=0.0
            )
        elif k == 'src_edge_type':
            v = pad_2d(
                [s[k] for s in data_unimol],
                pad_idx=0,
            )
        elif k == 'src_distance':
            v = pad_2d(
                [s[k] for s in data_unimol], pad_idx=0.0
            )
        elif k == 'src_tokens':
            v = pad_1d_tokens(
                [s[k] for s in data_unimol],
                pad_idx=0,
            )
        elif k == 'smi':
            v = [s[k] for s in data_unimol]
        elif k == 'ir':
            v = torch.cat([s[k] for s in data_unimol], dim=0)
        elif k == 'raman':
            v = torch.cat([s[k] for s in data_unimol], dim=0)
        elif k == 'uv':
            v = torch.cat([s[k] for s in data_unimol], dim=0)
        unimol_batch[k] = v
    pyg_batch = Batch.from_data_list(data_pyg)
    batch = {'unimol': unimol_batch, 'pyg': pyg_batch}
    return batch