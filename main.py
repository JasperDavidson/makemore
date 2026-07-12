import torch


def norm_log_likelihood_loss(
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


def main():
    # Word dataset to train off of
    words = open("names.txt", "r").read().splitlines()

    # Set up probability tensor conversion arrays for indexing/retrieval
    char_probs = torch.ones((27, 27))  # Model smoothing = 1
    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for i, s in enumerate(chars)}

    # Count the number of occurences of every bigram
    for word in words:
        chrs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ch1_idx = stoi[ch1]
            ch2_idx = stoi[ch2]

            char_probs[ch1_idx][ch2_idx] += 1

    # Normalize the bigram rows + calculate the negative normalized log likelihood loss
    torch.nn.functional.normalize(char_probs, p=1, dim=1, out=char_probs)
    loss = norm_log_likelihood_loss(words, stoi, char_probs)
    print("Log likelihood loss = ", loss)

    # Given a random seed, sample from the bigram probabilites to form a new word
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


if __name__ == "__main__":
    main()
