import torch


def stat_norm_log_likelihood_loss(
    words: list[str], stoi: dict[str, int], bigram_probs: torch.Tensor
) -> torch.Tensor:
    log_likelihood = torch.Tensor([0.0])
    n_likelihoods = 0

    for word in words:
        chrs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ch1_idx = stoi[ch1]
            ch2_idx = stoi[ch2]

            n_likelihoods += 1
            log_likelihood += torch.log(bigram_probs[ch1_idx][ch2_idx])

    return -(log_likelihood / n_likelihoods)


def stat_bigram(words: list[str], stoi: dict[str, int], itos: dict[int, str]):
    # Count the number of occurrences of every bigram and form probabilities
    char_probs = torch.ones((27, 27))  # Model smoothing = 1
    for word in words:
        chrs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ch1_idx = stoi[ch1]
            ch2_idx = stoi[ch2]

            char_probs[ch1_idx][ch2_idx] += 1

    # Normalize the bigram rows + calculate the negative normalized log likelihood loss
    torch.nn.functional.normalize(char_probs, p=1, dim=1, out=char_probs)
    loss = stat_norm_log_likelihood_loss(words, stoi, char_probs)
    print("Log likelihood loss = ", loss)

    # Given a random seed, sample from the bigram probabilities to form a new word
    new_word = []
    prev_char_idx = 0
    word_gen = torch.Generator()
    word_gen.manual_seed(42)

    while True:
        prev_char_idx = int(
            torch.multinomial(char_probs[prev_char_idx], 1, generator=word_gen).item()
        )
        prev_char = itos[prev_char_idx]

        if prev_char == ".":
            break

        new_word.append(prev_char)

    print("Generated: ", "".join(new_word))


def train(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
    # Hyperparameters
    learning_rate = 10

    # Create the input->target pair tensors
    input: list[int] = []
    target: list[int] = []

    for word in words:
        chrs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ch1_num = stoi[ch1]
            ch2_num = stoi[ch2]

            input.append(ch1_num)
            target.append(ch2_num)

    input_tensor = torch.tensor(input)

    one_hot_in = torch.nn.functional.one_hot(input_tensor, num_classes=27).float()
    weights = torch.randn(27, 27, requires_grad=True)
    loss = torch.inf
    while loss > 2.5:
        # Compute the first (and currently only) nn layer
        # n x 27 dot 27 x 27 (27 neurons) = n x 27
        out = one_hot_in @ weights

        # Compute softmax to get the probabilities
        prob = torch.nn.functional.softmax(out, dim=-1)

        # Compute the neg. log likelihood of the model output
        loss = -prob[torch.arange(len(target)), target].log().mean()

        # Backward pass (gradient descent)
        weights.grad = None
        loss.backward()
        assert weights.grad is not None
        weights.data += -learning_rate * weights.grad

        print("Log likelihood loss: ", loss.data)

    return weights


def infer_word(
    word_len: int, weights: torch.Tensor, stoi: dict[str, int], itos: dict[int, str]
):
    out_char = stoi["."]
    one_hot = torch.nn.functional.one_hot(
        torch.tensor(out_char), num_classes=27
    ).float()
    out_word = ""
    for _ in range(word_len):
        while True:
            out = one_hot @ weights
            prob = torch.nn.functional.softmax(out, dim=-1)

            out_char = int(torch.multinomial(prob, 1).item())

            if out_char != 0:
                break

        one_hot = torch.nn.functional.one_hot(
            torch.tensor(out_char), num_classes=27
        ).float()
        out_word += itos[out_char]

    print(out_word)


def main():
    # Word dataset to train off of
    words = open("names.txt", "r").read().splitlines()

    # Set conversion arrays for indexing/retrieval
    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for i, s in enumerate(chars)}

    # stat_bigram(words, stoi, itos)
    weights = train(words, stoi)
    infer_word(6, weights, stoi, itos)


if __name__ == "__main__":
    main()
