import torch


def find_mappings(rep: list[str], term: str) -> tuple[dict[int, str], dict[str, int]]:
    itos = {(i + 1): s for i, s in enumerate(rep)}
    stoi = {s: (i + 1) for i, s in enumerate(rep)}
    itos[0] = term
    stoi[term] = 0

    return (itos, stoi)


def create_data_splits(
    words: list[str], stoi: dict[str, int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    training, validation = [], []

    for word in words:
        context = [0] * block_size

        for ch in word + ".":
            ch_val = stoi[ch]
            training.append(context)
            validation.append(ch_val)
            context = context[1:] + [ch_val]

    training = torch.tensor(training)
    validation = torch.tensor(validation)

    return (training, validation)
