def find_mappings(rep: list[str], term: str) -> tuple[dict[int, str], dict[str, int]]:
    itos = {(i + 1): s for i, s in enumerate(rep)}
    stoi = {s: (i + 1) for i, s in enumerate(rep)}
    itos[0] = term
    stoi[term] = 0

    return (itos, stoi)
