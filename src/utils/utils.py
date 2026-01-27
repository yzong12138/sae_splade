import torch


def unpad(value, mask):
    """removes the padding tokens of the sequence (nD) to (n-1 D) matrix
    input value shape [bs, length, ...] or [bs_length]
    mask shape [bs, length]

    return unpad_values shape [total_nnz, ...] or [total_nnz] indices shape
    [total_nnz]
    """
    indices = torch.nonzero(mask.flatten()).flatten()  # shape [total_nnz]
    if value.dim() == 2:
        unpad_values = value.flatten()[indices]
    else:
        batch, seqlen, *rest = value.shape
        shape = batch * seqlen
        unpad_values = value.reshape(shape, *rest)[indices]
    return unpad_values, indices


def repad(unpad_values, indices, bs, length):
    """transform back the value back from n-1D back to nD
    input
    unpad_value shape [total_nnz, ...] or [total_nnz]
    indices [total_nnz]
    return
    value [bs, length, ...] or [bs, length]
    """
    if unpad_values.dim() == 1:
        output = torch.zeros(
            bs * length, dtype=unpad_values.dtype, device=unpad_values.device
        )
        output[indices] = unpad_values
        value = output.view(bs, length)
    else:
        _, *rest = unpad_values.shape
        output = torch.zeros(
            bs * length, *rest, dtype=unpad_values.dtype, device=unpad_values.device
        )
        output[indices] = unpad_values
        value = output.reshape(bs, length, *rest)
    return value
