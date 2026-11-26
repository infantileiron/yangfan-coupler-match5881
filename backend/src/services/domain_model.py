"""Sparse categorical logistic regression and shared holdout metrics."""

from collections import Counter
from dataclasses import dataclass
from math import exp, fsum, log, log1p, sqrt

from .domain_data import BenchRecord, Profile

Feature = tuple[str, ...]
# Each group stores active feature indices, trial count, and pass count.
Group = tuple[tuple[int, ...], int, int]


def encode(profile: Profile, crossed: bool) -> tuple[Feature, ...]:
    main_effects = (
        ("vehicle", profile.vehicle),
        ("head", profile.head),
        ("height_band", profile.height_band),
    )
    if crossed:
        return (*main_effects, ("cross", profile.vehicle, profile.head, profile.height_band))
    return main_effects


def sigmoid(score: float) -> float:
    if score >= 0:
        return 1 / (1 + exp(-score))
    odds = exp(score)
    return odds / (1 + odds)


@dataclass(frozen=True)
class LogisticModel:
    intercept: float
    coefficients: dict[Feature, float]
    crossed: bool

    def probability(self, profile: Profile) -> float:
        terms = encode(profile, self.crossed)
        score = self.intercept + fsum(self.coefficients.get(term, 0.0) for term in terms)
        return sigmoid(score)


def fit(records: list[BenchRecord], regularization: float, *, crossed: bool) -> LogisticModel:
    """Minimize mean Bernoulli log loss plus λ/2 times squared feature weights."""
    counts = Counter(record.profile for record in records)
    successes = Counter(record.profile for record in records if record.passed == 1)
    terms_by_profile = {profile: encode(profile, crossed) for profile in counts}
    vocabulary = sorted({term for terms in terms_by_profile.values() for term in terms})
    feature_indices = {term: i for i, term in enumerate(vocabulary, start=1)}
    groups = [
        ((0, *(feature_indices[term] for term in terms)), counts[profile], successes[profile])
        for profile, terms in terms_by_profile.items()
    ]
    total = len(records)
    passed = sum(successes.values())
    intercept = log(passed / (total - passed))
    weights = [intercept] + [0.0] * len(vocabulary)
    weights = _newton_fit(groups, total, weights, regularization)

    coefficients = {term: weights[index] for term, index in feature_indices.items()}
    return LogisticModel(weights[0], coefficients, crossed)


def _softplus(score: float) -> float:
    return max(score, 0.0) + log1p(exp(-abs(score)))


def _objective(groups: list[Group], total: int, weights: list[float], regularization: float) -> float:
    loss = 0.0
    for indices, count, passed in groups:
        score = fsum(weights[i] for i in indices)
        loss += (count - passed) * _softplus(score) + passed * _softplus(-score)
    return loss / total + 0.5 * regularization * fsum(weight * weight for weight in weights[1:])


def _cholesky_solve(hessian: list[list[float]], gradient: list[float]) -> list[float]:
    size = len(gradient)
    lower = [[0.0] * size for _ in gradient]
    for row in range(size):
        for column in range(row + 1):
            value = hessian[row][column] - fsum(lower[row][i] * lower[column][i] for i in range(column))
            lower[row][column] = sqrt(value) if row == column else value / lower[column][column]
    intermediate = [0.0] * size
    for row in range(size):
        value = gradient[row] - fsum(lower[row][i] * intermediate[i] for i in range(row))
        intermediate[row] = value / lower[row][row]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        value = intermediate[row] - fsum(lower[i][row] * solution[i] for i in range(row + 1, size))
        solution[row] = value / lower[row][row]
    return solution


def _newton_fit(groups: list[Group], total: int, weights: list[float], regularization: float) -> list[float]:
    # The largest design has 36 features plus its intercept, so an exact Hessian is small.
    # Ridge keeps it positive definite despite the redundant one-hot levels.
    for _ in range(50):
        gradient = [0.0] + [regularization * weight for weight in weights[1:]]
        hessian = [[0.0] * len(weights) for _ in weights]
        for index in range(1, len(weights)):
            hessian[index][index] = regularization
        for indices, count, passed in groups:
            score = fsum(weights[i] for i in indices)
            probability, failure_probability = sigmoid(score), sigmoid(-score)
            error = ((count - passed) * probability - passed * failure_probability) / total
            curvature = count * probability * failure_probability / total
            for row in indices:
                gradient[row] += error
                for column in indices:
                    hessian[row][column] += curvature
        if max(abs(component) for component in gradient) <= 1e-9:
            return weights

        direction = _cholesky_solve(hessian, gradient)
        decrease = fsum(component * delta for component, delta in zip(gradient, direction))
        current = _objective(groups, total, weights, regularization)
        for exponent in range(20):
            step = 0.5 ** exponent
            candidate = [weight - step * delta for weight, delta in zip(weights, direction)]
            if _objective(groups, total, candidate, regularization) <= current - 1e-4 * step * decrease + 1e-15:
                weights = candidate
                break
        else:
            break
    raise ValueError("车钩适配模型未收敛，请调高正则化强度后重试。")


@dataclass(frozen=True)
class HoldoutScores:
    log_loss: float
    accuracy: float


def evaluate(model: LogisticModel, records: list[BenchRecord]) -> HoldoutScores:
    predictions = [(model.probability(record.profile), record.passed) for record in records]
    epsilon = 1e-15
    losses = []
    for probability, passed in predictions:
        clipped = min(1 - epsilon, max(epsilon, probability))
        losses.append(-log(clipped) if passed else -log1p(-clipped))
    accuracy = sum((probability >= 0.5) == bool(passed) for probability, passed in predictions) / len(records)
    return HoldoutScores(fsum(losses) / len(records), accuracy)
