from math import ceil, log2
from typing import List, Tuple


def minimax(depth: int, index: int, is_max: bool,
            scores: List[int], target_depth: int,
            counter: List[int]) -> int:
  
    if depth == target_depth:
        counter[0] += 1
        return scores[index]

    left = minimax(depth + 1, index * 2, not is_max,
                   scores, target_depth, counter)
    right = minimax(depth + 1, index * 2 + 1, not is_max,
                    scores, target_depth, counter)

    if is_max:
        return max(left, right)
    return min(left, right)


def alpha_beta(depth: int, index: int, is_max: bool,
               scores: List[int], target_depth: int,
               alpha: float, beta: float,
               counter: List[int]) -> int:

    if depth == target_depth:
        counter[0] += 1
        return scores[index]

    if is_max:
        value = float("-inf")
        for child in range(2):
            value = max(
                value,
                alpha_beta(depth + 1, index * 2 + child, False,
                           scores, target_depth, alpha, beta, counter)
            )
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return int(value)

    value = float("inf")
    for child in range(2):
        value = min(
            value,
            alpha_beta(depth + 1, index * 2 + child, True,
                       scores, target_depth, alpha, beta, counter)
        )
        beta = min(beta, value)
        if alpha >= beta:
            break
    return int(value)


def tree_depth(number_of_leaves: int) -> int:
  
    return ceil(log2(number_of_leaves))


def prepare_scores(values: List[int]) -> List[int]:
  
    depth = tree_depth(len(values))
    required = 2 ** depth
    return values + [0] * (required - len(values))


def show_minimax_trace(scores: List[int]) -> None:
   
    current = scores[:]
    level_number = 1
    is_max = True

    print("\nMinimax trace (from the bottom level to the root):")
    print(f"Leaves: {current}")
    while len(current) > 1:
        next_level = []
        for pos in range(0, len(current), 2):
            pair = current[pos:pos + 2]
            next_level.append(max(pair) if is_max else min(pair))
        player = "MAX" if is_max else "MIN"
        print(f"{player} level {level_number}: {next_level}")
        current = next_level
        is_max = not is_max
        level_number += 1


def compare_order(scores: List[int], label: str) -> Tuple[int, int, int]:
    depth = int(log2(len(scores)))

    minimax_counter = [0]
    plain_value = minimax(0, 0, True, scores, depth, minimax_counter)

    alpha_beta_counter = [0]
    pruned_value = alpha_beta(
        0, 0, True, scores, depth,
        float("-inf"), float("inf"), alpha_beta_counter
    )

    print(f"\n{label}")
    print(f"Leaf scores: {scores}")
    print(f"Minimax value: {plain_value}")
    print(f"Minimax leaves evaluated: {minimax_counter[0]}")
    print(f"Alpha-Beta value: {pruned_value}")
    print(f"Alpha-Beta leaves evaluated: {alpha_beta_counter[0]}")
    print(f"Leaves saved: {minimax_counter[0] - alpha_beta_counter[0]}")

    # This check makes sure pruning has not changed the decision value.
    assert plain_value == pruned_value, "The algorithms returned different values."
    return plain_value, minimax_counter[0], alpha_beta_counter[0]


def main() -> None:
    supplied_scores = [3, 5, 2, 9, 0, 7]
    scores = prepare_scores(supplied_scores)
    depth = int(log2(len(scores)))

    print("LAB 5 - MINIMAX AND ALPHA-BETA PRUNING")
    print("--------------------------------------")
    print(f"Scores from the manual: {supplied_scores}")
    print(f"Scores after padding:    {scores}")
    print(f"Tree depth:              {depth}")

    show_minimax_trace(scores)

    print("\nComparison using the original left-to-right order:")
    original = compare_order(scores, "Original branch order")

    reversed_scores = list(reversed(scores))
    print("\nComparison after reversing the leaf order:")
    reordered = compare_order(reversed_scores, "Reversed branch order")

    print("\nFinal check:")
    print(f"Root value before reordering: {original[0]}")
    print(f"Root value after reordering:  {reordered[0]}")
    print("Both root values are equal:", original[0] == reordered[0])
    print("The number of Alpha-Beta leaf evaluations can change with ordering.")


if __name__ == "__main__":
    main()
