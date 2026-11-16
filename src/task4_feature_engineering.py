"""
Task 4: Advanced Feature Engineering Pipeline
Implements comprehensive feature engineering techniques including mathematical
transformations, categorical encoding, interaction features, and polynomial
features.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import warnings
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

warnings.filterwarnings("ignore")


class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for creating, transforming,
    and selecting features to improve model performance.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_catalog: dict[str, str] = {}
        self.encoders: dict[str, object] = {}
        self.original_features: list[str] = []
        self.engineered_features: list[str] = []
        self.feature_importance_scores: dict[str, dict[str, float]] = {}

    def create_mathematical_features(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        operations: list[str] | None = None,
    ) -> pd.DataFrame:
        ops = operations if operations else ["log", "sqrt", "square", "cube"]
        df = dataframe.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for op in ops:
                if op == "log":
                    min_val = cast(float, df[col].min())
                    shifted = df[col] - min_val + 1 if min_val <= 0 else df[col]
                    df[f"{col}_log"] = np.log(np.asarray(shifted, dtype=float))
                elif op == "sqrt":
                    df[f"{col}_sqrt"] = np.sqrt(
                        np.abs(np.asarray(df[col], dtype=float))
                    )
                elif op == "square":
                    df[f"{col}_square"] = np.asarray(df[col], dtype=float) ** 2
                elif op == "cube":
                    df[f"{col}_cube"] = np.asarray(df[col], dtype=float) ** 3
                else:
                    continue

                new_col = f"{col}_{op}"
                self.engineered_features.append(new_col)
                self.feature_catalog[new_col] = (
                    f"{op.capitalize()} transformation of {col}"
                )

        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: list[tuple[str, str]],
        operations: list[str] | None = None,
    ) -> pd.DataFrame:
        ops = operations if operations else ["multiply", "add", "subtract", "divide"]

        result = df.copy()

        for col1, col2 in column_pairs:
            if col1 not in result.columns or col2 not in result.columns:
                continue

            a = np.asarray(result[col1], dtype=float)
            b = np.asarray(result[col2], dtype=float)

            for op in ops:
                if op == "multiply":
                    name = f"{col1}_x_{col2}"
                    result[name] = a * b
                elif op == "add":
                    name = f"{col1}_plus_{col2}"
                    result[name] = a + b
                elif op == "subtract":
                    name = f"{col1}_minus_{col2}"
                    result[name] = a - b
                elif op == "divide":
                    name = f"{col1}_div_{col2}"
                    denom = np.where(b == 0, np.nan, b)
                    result[name] = np.nan_to_num(
                        a / denom, nan=0.0, posinf=0.0, neginf=0.0
                    )
                else:
                    continue

                self.engineered_features.append(name)
                self.feature_catalog[name] = f"Interaction: {col1} {op} {col2}"

        return result

    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_by: str,
        agg_columns: list[str],
        agg_functions: list[str] | None = None,
    ) -> pd.DataFrame:
        agg_f = (
            agg_functions if agg_functions else ["mean", "std", "min", "max", "count"]
        )

        result = df.copy()

        if group_by not in result.columns:
            return result

        valid_cols = [c for c in agg_columns if c in result.columns]
        if not valid_cols:
            return result

        grouped = result.groupby(group_by)

        for col in valid_cols:
            for func in agg_f:
                try:
                    agg_vals = grouped[col].transform(func)
                except Exception:
                    continue

                name = f"{col}_{func}_by_{group_by}"
                result[name] = np.asarray(agg_vals)
                self.engineered_features.append(name)
                self.feature_catalog[name] = (
                    f"{func.capitalize()} of {col} grouped by {group_by}"
                )

        return result

    def encode_onehot(
        self,
        df: pd.DataFrame,
        columns: list[str],
        drop_first: bool = False,
        max_categories: int = 10,
    ) -> pd.DataFrame:
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue

            col_data = result[col].astype(str)

            value_counts = col_data.value_counts()
            if len(value_counts) > max_categories:
                top_cats = value_counts.head(max_categories).index.tolist()
                col_data = col_data.where(col_data.isin(top_cats), "Other")

            dummies = pd.get_dummies(col_data, prefix=col, drop_first=drop_first)
            dummies = dummies.astype(int)

            result = pd.concat([result.drop(columns=[col]), dummies], axis=1)

            for c in dummies.columns:
                self.engineered_features.append(c)
                self.feature_catalog[c] = f"One-hot encoded {col}: {c}"

        return result

    def encode_label(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue

            encoder = LabelEncoder()
            encoded = encoder.fit_transform(result[col].astype(str))
            name = f"{col}_label"
            result[name] = encoded
            self.encoders[f"label_{col}"] = encoder
            self.engineered_features.append(name)
            self.feature_catalog[name] = f"Label encoded {col}"

        return result

    def encode_target(
        self,
        df: pd.DataFrame,
        columns: list[str],
        target: str,
        smoothing: float = 0.0,
    ) -> pd.DataFrame:
        result = df.copy()

        if target not in result.columns:
            return result

        global_mean = cast(float, result[target].mean())

        for col in columns:
            if col not in result.columns:
                continue

            stats = result.groupby(col)[target].agg(["mean", "count"])
            mean_series = cast(pd.Series, stats["mean"])
            count_series = cast(pd.Series, stats["count"])
            smoothed = (mean_series * count_series + global_mean * smoothing) / (
                count_series + smoothing
            )
            smoothed_series = cast(pd.Series, smoothed)

            name = f"{col}_target"
            mapped = (
                cast(pd.Series, result[col])
                .map(smoothed_series)
                .astype(float)
                .fillna(global_mean)
            )
            result[name] = mapped
            self.encoders[f"target_{col}"] = smoothed_series.to_dict()
            self.engineered_features.append(name)
            self.feature_catalog[name] = f"Target encoded {col}"

        return result

    def encode_frequency(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue

            freq = cast(
                dict[str, float],
                result[col].value_counts(normalize=True).to_dict(),
            )
            name = f"{col}_freq"
            result[name] = (
                cast(pd.Series, result[col])
                .map(lambda v: freq.get(v, 0.0))
                .astype(float)
            )
            self.encoders[f"freq_{col}"] = freq
            self.engineered_features.append(name)
            self.feature_catalog[name] = f"Frequency encoded {col}"

        return result

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        result = df.copy()

        valid = [c for c in columns if c in result.columns]
        if not valid:
            return result

        poly = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=False
        )
        poly_arr = poly.fit_transform(np.asarray(result[valid], dtype=float))
        names = poly.get_feature_names_out(valid)

        for i, name in enumerate(names):
            if name == "1" or name in valid:
                continue

            new_name = f"poly_{name.replace(' ', '_')}"
            result[new_name] = poly_arr[:, i]
            self.engineered_features.append(new_name)
            self.feature_catalog[new_name] = f"Polynomial feature: {name}"

        return result

    def select_features_univariate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        score_func=f_classif,
    ) -> tuple[list[str], np.ndarray]:
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)

        mask = cast(np.ndarray, selector.get_support())
        selected_features = cast(list[str], X.columns[mask].tolist())

        scores = cast(np.ndarray, selector.scores_)

        self.feature_importance_scores["univariate"] = cast(
            dict[str, float], dict(zip(X.columns, scores))
        )

        return selected_features, scores

    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        step: int = 1,
    ) -> list[str]:
        estimator = LogisticRegression(
            max_iter=1000, random_state=self.random_state, solver="lbfgs"
        )

        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features, X.shape[1]),
            step=step,
        )
        selector.fit(X, y)

        mask = cast(np.ndarray, selector.get_support())
        selected_features = cast(list[str], X.columns[mask].tolist())

        rankings = cast(dict[str, float], dict(zip(X.columns, selector.ranking_)))
        self.feature_importance_scores["rfe"] = rankings

        return selected_features

    def select_features_model_based(
        self, X: pd.DataFrame, y: pd.Series, n_features: int = 10
    ) -> tuple[list[str], np.ndarray]:
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)

        importances = cast(np.ndarray, rf.feature_importances_)

        indices = np.argsort(importances)[::-1]
        top_indices = indices[:n_features]
        selected_features = cast(list[str], X.columns[top_indices].tolist())

        self.feature_importance_scores["random_forest"] = cast(
            dict[str, float], dict(zip(X.columns, importances))
        )

        return selected_features, importances

    def get_feature_catalog(self) -> pd.DataFrame:
        catalog_df = pd.DataFrame(
            [
                {"Feature": k, "Description": v, "Type": "Engineered"}
                for k, v in self.feature_catalog.items()
            ]
        )
        return catalog_df

    def evaluate_feature_impact(
        self,
        X_original: pd.DataFrame,
        X_engineered: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> dict[str, float]:
        model = LogisticRegression(
            max_iter=1000, random_state=self.random_state, solver="lbfgs"
        )

        scores_original = cross_val_score(
            model, X_original, y, cv=cv, scoring="accuracy"
        )
        scores_engineered = cross_val_score(
            model, X_engineered, y, cv=cv, scoring="accuracy"
        )

        return {
            "original_mean": float(scores_original.mean()),
            "original_std": float(scores_original.std()),
            "engineered_mean": float(scores_engineered.mean()),
            "engineered_std": float(scores_engineered.std()),
            "improvement": float(scores_engineered.mean() - scores_original.mean()),
        }


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    n_categorical: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(random_state)

    data = {}
    for i in range(n_features):
        data[f"num_feature_{i + 1}"] = np.random.randn(n_samples) * 10 + 50

    categories_list = [
        ["A", "B", "C", "D"],
        ["X", "Y", "Z"],
        ["Low", "Medium", "High"],
        ["Type1", "Type2", "Type3", "Type4"],
    ]

    for i in range(n_categorical):
        categories = categories_list[i % len(categories_list)]
        data[f"cat_feature_{i + 1}"] = np.random.choice(categories, size=n_samples)

    data["group"] = np.random.choice(["Group1", "Group2", "Group3"], size=n_samples)

    df = pd.DataFrame(data)

    target = (
        0.5 * df["num_feature_1"]
        + 0.3 * df["num_feature_2"]
        + 0.2 * df["num_feature_1"] * df["num_feature_2"]
        + np.random.randn(n_samples) * 5
    )

    y = pd.Series(
        (target > target.median()).astype(int).values,
        index=df.index,
        name="target",
    )

    return df, y


def visualize_feature_importance(
    importance_scores: dict[str, float],
    top_n: int = 15,
    title: str = "Feature Importance",
    output_path: str = "feature_importance.png",
):
    sorted_features = sorted(
        importance_scores.items(), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    features, scores = zip(*sorted_features)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    plt.barh(range(len(features)), scores, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature importance plot saved to {output_path}")


def visualize_feature_correlations(
    df: pd.DataFrame,
    features: list[str],
    title: str = "Feature Correlation Matrix",
    output_path: str = "feature_correlations.png",
):
    valid_features = [f for f in features if f in df.columns]

    if not valid_features:
        print("No valid features found for correlation plot")
        return

    corr = df[valid_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Correlation heatmap saved to {output_path}")


def visualize_pairplot(
    df: pd.DataFrame,
    features: list[str],
    target: pd.Series | None = None,
    output_path: str = "feature_pairplot.png",
):
    features = features[:5]
    valid_features = [f for f in features if f in df.columns]

    if not valid_features:
        print("No valid features found for pair plot")
        return

    plot_df: pd.DataFrame = cast(pd.DataFrame, df[valid_features].copy())

    if target is not None:
        plot_df["Target"] = np.asarray(target)
        hue = "Target"
    else:
        hue = None

    pair_plot = sns.pairplot(
        plot_df, hue=hue, diag_kind="kde", plot_kws={"alpha": 0.6}, height=2.5
    )

    pair_plot.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Pair plot saved to {output_path}")


def visualize_polynomial_impact(
    X_original: np.ndarray,
    X_poly: np.ndarray,
    y: np.ndarray,
    feature_name: str = "Feature",
    output_path: str = "polynomial_features.png",
):
    X_arr = np.asarray(X_original, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(X_arr, y_arr, alpha=0.5, s=20)
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Target")
    axes[0].set_title("Original Feature vs Target")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(X_arr, y_arr, alpha=0.5, s=20, label="Data")

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures as PF

    poly = PF(degree=2)
    X_poly_fit = poly.fit_transform(X_arr.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly_fit, y_arr)

    X_plot = np.linspace(float(X_arr.min()), float(X_arr.max()), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_pred = model.predict(X_plot_poly)

    axes[1].plot(X_plot, y_pred, "r-", linewidth=2, label="Polynomial Fit")
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel("Target")
    axes[1].set_title("Polynomial Features (Degree 2)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Polynomial feature visualization saved to {output_path}")


def visualize_encoding_comparison(
    performance_dict: dict[str, float],
    output_path: str = "encoding_comparison.png",
):
    methods = list(performance_dict.keys())
    scores = list(performance_dict.values())

    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = plt.bar(
        range(len(methods)),
        scores,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.ylabel("Model Accuracy")
    plt.title("Performance Comparison: Different Encoding Techniques")
    plt.ylim(0, max(scores) * 1.15)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Encoding comparison plot saved to {output_path}")


def visualize_feature_network(
    interaction_pairs: list[tuple[str, str]],
    importance_scores: dict[str, float],
    output_path: str = "feature_network.png",
):
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        print("NetworkX not installed. Skipping network visualization.")
        return

    G = nx.Graph()

    all_features = set()
    for f1, f2 in interaction_pairs:
        all_features.add(f1)
        all_features.add(f2)

    for feature in all_features:
        importance = importance_scores.get(feature, 0.5)
        G.add_node(feature, importance=importance)

    for f1, f2 in interaction_pairs:
        G.add_edge(f1, f2)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    node_sizes = [G.nodes[node].get("importance", 0.5) * 2000 for node in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="lightblue",
        edgecolors="black",
        linewidths=2,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title("Feature Interaction Network", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature network visualization saved to {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Task 4: Advanced Feature Engineering Pipeline")
    print("=" * 80)

    print("\n1. Creating sample dataset...")
    df, y = create_sample_dataset(n_samples=1000, n_features=5, n_categorical=2)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    print("\n2. Initializing Feature Engineering Pipeline...")
    pipeline = FeatureEngineeringPipeline(random_state=42)

    original_numerical = ["num_feature_1", "num_feature_2", "num_feature_3"]
    original_categorical = ["cat_feature_1", "cat_feature_2"]

    print("\n3. Creating mathematical features...")
    df = pipeline.create_mathematical_features(
        df,
        columns=original_numerical[:2],
        operations=["log", "sqrt", "square"],
    )
    print(f"Features after mathematical transformations: {df.shape[1]}")

    print("\n4. Creating interaction features...")
    interaction_pairs = [
        ("num_feature_1", "num_feature_2"),
        ("num_feature_1", "num_feature_3"),
        ("num_feature_2", "num_feature_3"),
    ]
    df = pipeline.create_interaction_features(
        df,
        column_pairs=interaction_pairs,
        operations=["multiply", "add", "divide"],
    )
    print(f"Features after interactions: {df.shape[1]}")

    print("\n5. Creating aggregation features...")
    df = pipeline.create_aggregation_features(
        df,
        group_by="group",
        agg_columns=original_numerical[:2],
        agg_functions=["mean", "std", "max"],
    )
    print(f"Features after aggregations: {df.shape[1]}")

    print("\n6. Encoding categorical features...")

    df["target_temp"] = np.asarray(y)
    df = pipeline.encode_target(
        df,
        columns=["cat_feature_1"],
        target="target_temp",
        smoothing=10.0,
    )
    df = pipeline.encode_target(
        df,
        columns=["cat_feature_2"],
        target="target_temp",
        smoothing=10.0,
    )
    df = df.drop(columns=["target_temp"])

    df = pipeline.encode_frequency(df, columns=["group"])
    df = pipeline.encode_label(df, columns=original_categorical)
    df = pipeline.encode_onehot(
        df, columns=original_categorical + ["group"], drop_first=True
    )

    print(f"Features after encoding: {df.shape[1]}")

    print("\n7. Creating polynomial features...")
    df = pipeline.create_polynomial_features(
        df,
        columns=original_numerical[:2],
        degree=2,
        include_bias=False,
    )
    print(f"Total features after all engineering: {df.shape[1]}")

    print("\n8. Performing feature selection...")

    selected_univariate, scores_univariate = pipeline.select_features_univariate(
        df, y, k=20, score_func=f_classif
    )
    print(f"Top 20 features (univariate): {len(selected_univariate)}")

    selected_rf, scores_rf = pipeline.select_features_model_based(df, y, n_features=20)
    print(f"Top 20 features (Random Forest): {len(selected_rf)}")

    selected_rfe = pipeline.select_features_rfe(df, y, n_features=20, step=5)
    print(f"Top 20 features (RFE): {len(selected_rfe)}")

    print("\n9. Generating feature catalog...")
    catalog = pipeline.get_feature_catalog()
    print(f"Total engineered features cataloged: {len(catalog)}")
    print("\nSample catalog entries:")
    print(catalog.head(10).to_string(index=False))

    print("\n10. Evaluating feature engineering impact...")
    X_original = cast(pd.DataFrame, df[original_numerical].copy())
    X_engineered = cast(pd.DataFrame, df[selected_rf].copy())

    impact = pipeline.evaluate_feature_impact(X_original, X_engineered, y, cv=5)
    print(
        f"\nOriginal features accuracy: "
        f"{impact['original_mean']:.4f} "
        f"(+/- {impact['original_std']:.4f})"
    )
    print(
        f"Engineered features accuracy: "
        f"{impact['engineered_mean']:.4f} "
        f"(+/- {impact['engineered_std']:.4f})"
    )
    print(f"Improvement: {impact['improvement']:.4f}")

    print("\n11. Generating visualizations...")

    visualize_feature_importance(
        pipeline.feature_importance_scores["random_forest"],
        top_n=15,
        title="Top 15 Features by Random Forest Importance",
        output_path="task4_feature_importance.png",
    )

    visualize_feature_importance(
        pipeline.feature_importance_scores["univariate"],
        top_n=15,
        title="Top 15 Features by Univariate F-Score",
        output_path="task4_univariate_importance.png",
    )

    visualize_feature_correlations(
        df,
        features=selected_rf[:20],
        title="Correlation Matrix of Top 20 Features",
        output_path="task4_feature_correlations.png",
    )

    interaction_feature_names = [f"{col1}_x_{col2}" for col1, col2 in interaction_pairs]
    visualize_pairplot(
        df,
        features=interaction_feature_names,
        target=y,
        output_path="task4_interaction_pairplot.png",
    )

    visualize_polynomial_impact(
        np.asarray(df["num_feature_1"], dtype=float),
        np.asarray(df[["num_feature_1", "poly_num_feature_1^2"]], dtype=float),
        np.asarray(y, dtype=float),
        feature_name="num_feature_1",
        output_path="task4_polynomial_impact.png",
    )

    print("\n12. Comparing encoding techniques...")
    encoding_performance: dict[str, float] = {}

    df_temp, y_temp = create_sample_dataset(n_samples=1000, random_state=42)

    df_label = df_temp.copy()
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df_label["cat_feature_1_encoded"] = le1.fit_transform(df_label["cat_feature_1"])
    df_label["cat_feature_2_encoded"] = le2.fit_transform(df_label["cat_feature_2"])
    X_label = df_label[
        [
            "num_feature_1",
            "num_feature_2",
            "num_feature_3",
            "cat_feature_1_encoded",
            "cat_feature_2_encoded",
        ]
    ]
    model = LogisticRegression(max_iter=1000, random_state=42)
    encoding_performance["Label Encoding"] = float(
        cross_val_score(model, X_label, y_temp, cv=5).mean()
    )

    df_onehot = pd.get_dummies(
        df_temp,
        columns=["cat_feature_1", "cat_feature_2"],
        drop_first=True,
    )
    X_onehot = df_onehot.drop(columns=["group"])
    encoding_performance["One-Hot Encoding"] = float(
        cross_val_score(model, X_onehot, y_temp, cv=5).mean()
    )

    freq1 = cast(
        dict[str, float],
        df_temp["cat_feature_1"].value_counts(normalize=True).to_dict(),
    )
    freq2 = cast(
        dict[str, float],
        df_temp["cat_feature_2"].value_counts(normalize=True).to_dict(),
    )
    df_freq = df_temp.copy()
    df_freq["cat_feature_1_freq"] = (
        cast(pd.Series, df_freq["cat_feature_1"])
        .map(lambda v: freq1.get(v, 0.0))
        .astype(float)
    )
    df_freq["cat_feature_2_freq"] = (
        cast(pd.Series, df_freq["cat_feature_2"])
        .map(lambda v: freq2.get(v, 0.0))
        .astype(float)
    )
    X_freq = df_freq[
        [
            "num_feature_1",
            "num_feature_2",
            "num_feature_3",
            "cat_feature_1_freq",
            "cat_feature_2_freq",
        ]
    ]
    encoding_performance["Frequency Encoding"] = float(
        cross_val_score(model, X_freq, y_temp, cv=5).mean()
    )

    print("\nEncoding performance comparison:")
    for method, score in encoding_performance.items():
        print(f"  {method}: {score:.4f}")

    visualize_encoding_comparison(
        encoding_performance,
        output_path="task4_encoding_comparison.png",
    )

    visualize_feature_network(
        interaction_pairs,
        pipeline.feature_importance_scores["random_forest"],
        output_path="task4_feature_network.png",
    )

    print("\n" + "=" * 80)
    print("Feature Engineering Pipeline Complete!")
    print("=" * 80)
    print(f"\nOriginal features: {len(original_numerical) + len(original_categorical)}")
    print(f"Engineered features: {df.shape[1]}")
    print(f"Top selected features: {len(selected_rf)}")
    print(f"Performance improvement: {impact['improvement']:.4f}")
    print("\nGenerated visualizations:")
    print("  - task4_feature_importance.png")
    print("  - task4_univariate_importance.png")
    print("  - task4_feature_correlations.png")
    print("  - task4_interaction_pairplot.png")
    print("  - task4_polynomial_impact.png")
    print("  - task4_encoding_comparison.png")
    print("  - task4_feature_network.png")
