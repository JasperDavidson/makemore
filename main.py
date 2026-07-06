import torch


def main():
    words = open("names.txt", "r").read().splitlines()

    char_probs = torch.zeros((27, 27))
    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for i, s in enumerate(chars)}

    for word in words:
        chrs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            ch1_idx = stoi[ch1]
            ch2_idx = stoi[ch2]

            char_probs[ch1_idx][ch2_idx] += 1

    normalized_char_probs = torch.nn.functional.normalize(char_probs, p=1, dim=1)

    new_word = []
    prev_char_idx = 0
    word_gen = torch.Generator()
    word_gen.manual_seed(42)

    while True:
        prev_char_idx = int(
            torch.multinomial(
                normalized_char_probs[prev_char_idx], 1, generator=word_gen
            ).item()
        )
        prev_char = itos[prev_char_idx]

        if prev_char == ".":
            break

        new_word.append(prev_char)

    print("Generated: ", "".join(new_word))


if __name__ == "__main__":
    main()
