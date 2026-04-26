"""
Buffer Size Sensitivity Analysis for ASRRL IDS Framework.

Varies the Novelty Candidate Buffer size across [50, 100, 200, 500, 1000]
and measures F1 score on CSE-CIC-IDS-2018 and UNSW-NB15.

Saves results to results/enhanced/table_buffer_sensitivity.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# Import framework components from experiments.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (
    FEATURE_NAMES, Action, Z3ConstraintManager,
    DBSCANPatternDetector, SymbolicShieldAgent, _generate
)


def run_with_buffer_size(n, ds, buffer_size, epochs=5, seed=42):
    """Run ASRRL with a specific DBSCAN novelty-candidate buffer size.

    The buffer_size controls DBSCANPatternDetector's maxlen, i.e. how many
    misclassified samples are retained for novel-pattern clustering.
    """
    np.random.seed(seed)
    df = _generate(n, ds)
    split = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_NAMES])
    y_train = train_df["label"].values
    X_test = scaler.transform(test_df[FEATURE_NAMES])
    y_test = test_df["label"].values

    # Decision tree
    dt = DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=20,
        class_weight="balanced", random_state=seed
    )
    dt.fit(X_train, y_train)

    # Z3 constraints
    cm = Z3ConstraintManager()
    cm.extract_from_tree(dt, epoch=0)

    # DBSCAN detector with the varied buffer_size
    dbscan = DBSCANPatternDetector(eps=1.5, min_samples=5, buffer_size=buffer_size)

    # RL agent
    agent = SymbolicShieldAgent(n_actions=3, lr=0.10, gamma=0.90, eps_start=0.30)
    agent.dt_model = dt

    # Training loop (reduced epochs for speed)
    for epoch in range(epochs):
        for i in range(len(X_train)):
            state = X_train[i]
            true_label = int(y_train[i])

            action, shielded = agent.act(state, cm, training=True)
            r = agent.reward(action, true_label, shielded)

            next_state = X_train[(i + 1) % len(X_train)]
            done = (i == len(X_train) - 1)
            agent.update(state, action, r, next_state, done)

            correct_action = Action(true_label) if true_label < 2 else Action.UNKNOWN
            dbscan.add(state, action, correct_action)

        # Novel-pattern detection and Z3 injection
        if epoch % 2 == 0 and epoch > 0:
            novel = dbscan.detect()
            for pattern in novel:
                path = [
                    (fi, float(val * 1.1), "<=", None)
                    for fi, val in enumerate(pattern) if fi < len(FEATURE_NAMES)
                ]
                path += [
                    (fi, float(val * 0.9), ">", None)
                    for fi, val in enumerate(pattern) if fi < len(FEATURE_NAMES)
                ]
                cm.add_constraint_from_path(path, Action.BLOCK)

    # Evaluation
    preds = []
    for i in range(len(X_test)):
        state = X_test[i]
        action, _ = agent.act(state, cm, training=False)
        preds.append(1 if action == Action.BLOCK else 0)

    f1 = f1_score(y_test, np.array(preds), zero_division=0)
    return f1


def main():
    buffer_sizes = [50, 100, 200, 500, 1000]
    datasets = ["CSE", "UNSW"]
    dataset_labels = {"CSE": "CSE-CIC-IDS-2018", "UNSW": "UNSW-NB15"}

    n_samples = 3000   # enough to show trends; fast enough to complete quickly
    epochs = 5

    results = []

    for bs in buffer_sizes:
        for ds in datasets:
            print(f"Running: buffer_size={bs}, dataset={dataset_labels[ds]} ...", flush=True)
            f1 = run_with_buffer_size(n_samples, ds, buffer_size=bs, epochs=epochs)
            results.append({
                "buffer_size": bs,
                "dataset": dataset_labels[ds],
                "f1_score": round(f1, 4),
            })
            print(f"  F1 = {f1:.4f}")

    # Save CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "enhanced")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "table_buffer_sensitivity.csv")
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
