"""Benchmarks that apply the same protocol to every set of word vectors."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr

from .analogies import _space, evaluate_analogies, most_similar

SEMANTIC_CATEGORIES = frozenset({
    "capital-common-countries", "capital-world", "currency", "city-in-state", "family",
})
CONTROL_WORDS = ("king", "france", "computer", "good", "january", "run")
ANALOGY_METHODS = ("3cosadd", "3cosmul")

THEMATIC_GROUPS: dict[str, tuple[str, ...]] = {
    "countries": (
        "france", "germany", "italy", "spain", "portugal", "england", "scotland", "ireland", "poland",
        "russia", "china", "japan", "india", "korea", "vietnam", "thailand", "indonesia", "iran", "iraq",
        "egypt", "israel", "turkey", "greece", "sweden", "norway", "denmark", "finland", "austria",
        "switzerland", "belgium", "netherlands", "canada", "mexico", "brazil", "argentina", "chile",
        "peru", "colombia", "cuba", "australia", "ukraine", "romania", "hungary", "bulgaria", "serbia",
        "croatia", "nigeria", "kenya", "ethiopia", "morocco", "algeria", "pakistan", "afghanistan",
    ),
    "cities": (
        "paris", "berlin", "rome", "madrid", "lisbon", "london", "dublin", "warsaw", "moscow", "beijing",
        "tokyo", "delhi", "seoul", "bangkok", "jakarta", "tehran", "baghdad", "cairo", "jerusalem",
        "athens", "stockholm", "oslo", "copenhagen", "helsinki", "vienna", "zurich", "brussels",
        "amsterdam", "toronto", "montreal", "chicago", "boston", "houston", "seattle", "denver",
        "atlanta", "detroit", "philadelphia", "dallas", "miami", "sydney", "melbourne", "budapest",
        "prague", "bucharest", "belgrade", "lagos", "nairobi", "manila", "shanghai", "glasgow",
        "manchester", "liverpool", "birmingham",
    ),
    "animals": (
        "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "rabbit", "mouse", "rat", "lion", "tiger",
        "bear", "wolf", "fox", "deer", "elephant", "monkey", "snake", "lizard", "frog", "fish", "shark",
        "whale", "dolphin", "bird", "eagle", "hawk", "owl", "duck", "goose", "chicken", "crow", "swan",
        "bee", "ant", "spider", "butterfly", "fly", "beetle", "turtle", "crocodile", "camel", "zebra",
        "giraffe", "squirrel", "bat", "seal", "octopus", "salmon", "donkey", "leopard",
    ),
    "numbers": (
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand", "million", "billion", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "half", "quarter", "double", "triple", "dozen",
        "zero", "twentieth", "fiftieth", "hundreds", "thousands",
    ),
    "family": (
        "man", "woman", "boy", "girl", "child", "father", "mother", "son", "daughter", "brother",
        "sister", "husband", "wife", "uncle", "aunt", "cousin", "nephew", "niece", "grandfather",
        "grandmother", "parents", "children", "baby", "king", "queen", "prince", "princess", "lord",
        "lady", "emperor", "duke", "bride", "widow", "friend", "neighbor", "men", "women", "boys",
        "girls", "twins", "heir", "orphan", "family", "mom", "dad", "teenager", "adult", "elder",
        "kid", "gentleman",
    ),
    "professions": (
        "doctor", "nurse", "teacher", "professor", "student", "lawyer", "judge", "police", "soldier",
        "officer", "pilot", "driver", "sailor", "farmer", "miner", "carpenter", "engineer", "scientist",
        "chemist", "physicist", "architect", "painter", "sculptor", "writer", "poet", "novelist",
        "journalist", "editor", "actor", "actress", "singer", "musician", "composer", "dancer",
        "director", "producer", "athlete", "coach", "referee", "banker", "merchant", "trader",
        "politician", "senator", "mayor", "governor", "president", "priest", "bishop", "monk",
        "chef", "baker", "waiter", "clerk",
    ),
    "verbs": (
        "walk", "run", "jump", "swim", "fly", "sit", "stand", "sleep", "eat", "drink", "cook", "write",
        "read", "speak", "talk", "listen", "see", "look", "watch", "hear", "think", "know", "learn",
        "teach", "play", "work", "build", "make", "break", "open", "close", "buy", "sell", "pay",
        "give", "take", "carry", "throw", "catch", "push", "pull", "fight", "win", "lose", "love",
        "hate", "help", "begin", "finish", "travel", "drive", "ride", "sing", "dance", "laugh",
    ),
    "body": (
        "head", "face", "eye", "ear", "nose", "mouth", "lip", "tooth", "tongue", "neck", "shoulder",
        "arm", "elbow", "hand", "finger", "thumb", "chest", "heart", "lung", "stomach", "back", "spine",
        "hip", "leg", "knee", "foot", "toe", "skin", "bone", "muscle", "blood", "brain", "hair",
        "beard", "throat", "wrist", "ankle", "nerve", "kidney", "liver", "skull", "palm", "heel",
        "forehead", "cheek", "chin", "eyebrow", "nail", "rib", "vein",
    ),
    "food": (
        "bread", "butter", "cheese", "milk", "egg", "meat", "beef", "pork", "lamb", "chicken", "fish",
        "rice", "pasta", "soup", "salad", "sugar", "salt", "pepper", "honey", "chocolate", "cake",
        "pie", "cookie", "apple", "banana", "orange", "lemon", "grape", "cherry", "strawberry",
        "peach", "pear", "tomato", "potato", "onion", "carrot", "corn", "bean", "wheat", "coffee",
        "tea", "wine", "beer", "water", "juice", "oil", "flour", "sauce", "cream", "nut", "olive",
    ),
    "time": (
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "spring", "summer", "autumn", "winter", "morning", "afternoon", "evening",
        "night", "noon", "midnight", "dawn", "dusk", "second", "minute", "hour", "day", "week", "month",
        "year", "decade", "century", "millennium", "today", "tomorrow", "yesterday", "weekend",
        "holiday", "season", "era", "age", "moment",
    ),
}


def load_similarity_pairs(path: str) -> list[tuple[str, str, float]]:
    """Read a tab separated word similarity file such as WordSim-353 or SimLex-999."""
    pairs: list[tuple[str, str, float]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            left, right, score = line.rstrip("\n").split("\t")[:3]
            pairs.append((left.lower(), right.lower(), float(score)))
    return pairs


def load_analogy_records(path: str) -> list[tuple[str, str, str, str, str]]:
    """Read questions-words.txt as (category, a, b, c, d) records in lowercase."""
    records: list[tuple[str, str, str, str, str]] = []
    category = ""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(":"):
                category = line[1:].strip()
            elif line.strip():
                a, b, c, d = line.lower().split()
                records.append((category, a, b, c, d))
    return records


def spearman_similarity(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    pairs: Iterable[tuple[str, str, float]],
) -> dict[str, float | int]:
    """Correlate cosine similarities with human scores over in-vocabulary pairs."""
    matrix = np.asarray(vectors, dtype=np.float64)
    predicted: list[float] = []
    human: list[float] = []
    total = 0
    for left, right, score in pairs:
        total += 1
        if left not in vocabulary or right not in vocabulary:
            continue
        first, second = matrix[vocabulary[left]], matrix[vocabulary[right]]
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        predicted.append(float(first @ second / denominator) if denominator else 0.0)
        human.append(score)
    correlation = float(spearmanr(predicted, human).statistic) if len(predicted) > 1 else float("nan")
    return {
        "spearman": correlation,
        "pairs": len(predicted),
        "total": total,
        "coverage": len(predicted) / total if total else 0.0,
    }


def shared_vocabulary(
    ranked_words: Iterable[str],
    *vocabularies: Mapping[str, int],
    size: int = 30_000,
    exclude: Iterable[str] = ("<unk>",),
) -> list[str]:
    """Keep the most frequent ranked words that every vocabulary contains."""
    if size <= 0:
        raise ValueError("size must be positive")
    excluded = set(exclude)
    shared: list[str] = []
    for word in ranked_words:
        if word in excluded or any(word not in vocabulary for vocabulary in vocabularies):
            continue
        shared.append(word)
        if len(shared) == size:
            break
    return shared


def restrict_space(
    vectors: ArrayLike, vocabulary: Mapping[str, int], words: Sequence[str]
) -> tuple[NDArray[np.float64], dict[str, int]]:
    """Return the rows of the given words with a vocabulary re-indexed from zero."""
    matrix = np.asarray(vectors, dtype=np.float64)
    rows = np.asarray([vocabulary[word] for word in words], dtype=np.int64)
    return matrix[rows], {word: index for index, word in enumerate(words)}


def _summarize(
    records: Sequence[Sequence[str]], vectors: ArrayLike, vocabulary: Mapping[str, int], method: str
) -> dict[str, float | int]:
    metrics = evaluate_analogies(vectors, vocabulary, [record[1:] for record in records], method=method)
    metrics["correct"] = int(round(metrics["accuracy"] * metrics["evaluated"]))
    return metrics


def evaluate_by_category(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    records: Sequence[tuple[str, str, str, str, str]],
    *,
    methods: Sequence[str] = ANALOGY_METHODS,
    semantic_categories: Iterable[str] = SEMANTIC_CATEGORIES,
) -> pd.DataFrame:
    """Report coverage and accuracy per category plus semantic, syntactic and total rows."""
    semantic = set(semantic_categories)
    categories = list(dict.fromkeys(record[0] for record in records))
    groups: dict[str, list[tuple[str, str, str, str, str]]] = {
        category: [record for record in records if record[0] == category] for category in categories
    }
    groups["semantic"] = [record for record in records if record[0] in semantic]
    groups["syntactic"] = [record for record in records if record[0] not in semantic]
    groups["total"] = list(records)
    rows = []
    for method in methods:
        for name, subset in groups.items():
            metrics = _summarize(subset, vectors, vocabulary, method)
            rows.append({
                "method": method,
                "category": name,
                "kind": "aggregate" if name in {"semantic", "syntactic", "total"} else "category",
                "total": metrics["total"],
                "evaluated": metrics["evaluated"],
                "coverage": metrics["coverage"],
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "mrr": metrics["mrr"],
            })
    return pd.DataFrame(rows)


def epoch_evaluation(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    pairs: Sequence[tuple[str, str, float]],
    records: Sequence[tuple[str, str, str, str, str]],
    *,
    control_words: Sequence[str] = CONTROL_WORDS,
    neighbors: int = 5,
) -> dict[str, object]:
    """Compute the per epoch metrics used to compare training runs."""
    semantic = _summarize([r for r in records if r[0] in SEMANTIC_CATEGORIES], vectors, vocabulary, "3cosadd")
    syntactic = _summarize([r for r in records if r[0] not in SEMANTIC_CATEGORIES], vectors, vocabulary, "3cosadd")
    evaluated = semantic["evaluated"] + syntactic["evaluated"]
    correct = semantic["correct"] + syntactic["correct"]
    return {
        "wordsim_spearman": spearman_similarity(vectors, vocabulary, pairs)["spearman"],
        "semantic_accuracy": semantic["accuracy"],
        "syntactic_accuracy": syntactic["accuracy"],
        "total_accuracy": correct / evaluated if evaluated else 0.0,
        "neighbors": {
            word: most_similar(vectors, vocabulary, word, top_k=neighbors)
            for word in control_words if word in vocabulary
        },
    }


def difference_parallelism(
    vectors: ArrayLike,
    vocabulary: Mapping[str, int],
    records: Sequence[tuple[str, str, str, str, str]],
    *,
    max_pairs: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Average cosine between difference vectors of one category versus random word pairs."""
    normalized, words, indices = _space(vectors, vocabulary)
    rows_of = {word: normalized[index] for word, index in zip(words, indices)}
    rng = np.random.default_rng(seed)
    rows = []
    for category in dict.fromkeys(record[0] for record in records):
        pairs = {
            pair
            for record in records if record[0] == category
            for pair in ((record[1], record[2]), (record[3], record[4]))
            if pair[0] in rows_of and pair[1] in rows_of
        }
        ordered = sorted(pairs)
        if len(ordered) > max_pairs:
            ordered = [ordered[i] for i in rng.choice(len(ordered), max_pairs, replace=False)]
        if len(ordered) < 2:
            rows.append({"category": category, "pairs": len(ordered), "mean_cosine": float("nan"),
                         "random_mean_cosine": float("nan")})
            continue
        differences = np.stack([rows_of[b] - rows_of[a] for a, b in ordered])
        random_left = rng.integers(0, len(words), len(ordered))
        random_right = rng.integers(0, len(words), len(ordered))
        baseline = normalized[indices[random_right]] - normalized[indices[random_left]]
        rows.append({
            "category": category,
            "pairs": len(ordered),
            "mean_cosine": _mean_pairwise_cosine(differences),
            "random_mean_cosine": _mean_pairwise_cosine(baseline),
        })
    return pd.DataFrame(rows)


def _mean_pairwise_cosine(differences: NDArray[np.float64]) -> float:
    norms = np.linalg.norm(differences, axis=1, keepdims=True)
    unit = differences[norms[:, 0] > 0] / norms[norms[:, 0] > 0]
    count = len(unit)
    if count < 2:
        return float("nan")
    return float((np.linalg.norm(unit.sum(axis=0)) ** 2 - count) / (count * (count - 1)))


def select_thematic_words(
    available: Iterable[str],
    groups: Mapping[str, Sequence[str]] = THEMATIC_GROUPS,
    per_group: int = 50,
) -> tuple[list[str], list[str]]:
    """Pick up to per_group available words from each group, without repeats across groups."""
    allowed = set(available)
    words: list[str] = []
    labels: list[str] = []
    used: set[str] = set()
    for group, candidates in groups.items():
        taken = 0
        for word in candidates:
            if taken == per_group:
                break
            if word in allowed and word not in used:
                used.add(word)
                words.append(word)
                labels.append(group)
                taken += 1
    return words, labels
