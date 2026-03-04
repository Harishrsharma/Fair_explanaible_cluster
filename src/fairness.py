# fairness.py

import numpy as np


def fairlet_decomposition(X, sensitive, p=1, q=1):

    reds = np.where(sensitive == 1)[0].tolist()
    blues = np.where(sensitive == 0)[0].tolist()

    fairlets = []

    while len(reds) >= q and len(blues) >= p:
        r = [reds.pop() for _ in range(q)]
        b = [blues.pop() for _ in range(p)]
        fairlets.append(r + b)

    while len(reds) > 0 and len(blues) > 0:
        fairlets.append([reds.pop(), blues.pop()])

    return fairlets


def assign_labels_from_fairlets(fairlets, center_labels, n):

    labels = np.zeros(n, dtype=int)

    for fl, lab in zip(fairlets, center_labels):
        for idx in fl:
            labels[idx] = lab

    return labels


def fairness_metrics(labels, sensitive, p=1, q=1):

    K = len(np.unique(labels))
    balances = []
    violations = 0

    global_ratio = sensitive.mean()
    dp_gaps = []

    for k in range(K):

        mask = labels == k
        group = sensitive[mask]

        n0 = np.sum(group == 0)
        n1 = np.sum(group == 1)

        if max(n0, n1) == 0:
            continue

        balance = min(n0, n1) / max(n0, n1)
        balances.append(balance)

        if balance < (p / q):
            violations += 1

        cluster_ratio = group.mean()
        dp_gaps.append(abs(cluster_ratio - global_ratio))

    return {
        "avg_balance": np.mean(balances),
        "violation_rate": violations / K,
        "avg_dp_gap": np.mean(dp_gaps)
    }
