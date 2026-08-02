import torch
import random


class DataSet:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y


class DataSets:
    def __init__(self, training: DataSet, validation: DataSet, test: DataSet):
        self.training = training
        self.validation = validation
        self.test = test


def find_mappings(rep: list[str], term: str) -> tuple[dict[int, str], dict[str, int]]:
    itos = {(i + 1): s for i, s in enumerate(rep)}
    stoi = {s: (i + 1) for i, s in enumerate(rep)}
    itos[0] = term
    stoi[term] = 0

    return (itos, stoi)


def create_data_splits(
    words: list[str], stoi: dict[str, int], block_size: int
) -> DataSet:
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

    return DataSet(training, validation)


def compute_data_sets(block_size: int) -> DataSets:
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    _, stoi = find_mappings(chars, ".")

    random.shuffle(words)
    train_split = int(len(words) * 0.8)
    val_split = int(len(words) * 0.9)
    test_split = len(words)

    training_set = create_data_splits(words[0:train_split], stoi, block_size)
    validation_set = create_data_splits(words[train_split:val_split], stoi, block_size)
    test_set = create_data_splits(words[val_split:test_split], stoi, block_size)

    return DataSets(training_set, validation_set, test_set)
