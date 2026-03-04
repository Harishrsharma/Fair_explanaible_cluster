# explainability.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def explainability_metrics(X, labels, max_depth, random_state, test_size=0.3):

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels,
        test_size=test_size,
        random_state=random_state
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    clf.fit(X_train, y_train)

    fidelity = clf.score(X_test, y_test)

    return {
        "tree_fidelity": fidelity,
        "tree_depth": clf.get_depth(),
        "tree_leaves": clf.get_n_leaves()
    }
