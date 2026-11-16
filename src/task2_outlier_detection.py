"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import warnings
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.

        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        # Explicit annotation so Pyright narrows the value type to DataFrame
        self.outlier_masks: dict[str, pd.DataFrame] = {}
        self.treated_datasets: dict[str, pd.DataFrame] = {}

    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.

        Parameters:
            threshold (float): Z-score threshold for outlier detection

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        mask: pd.DataFrame = pd.DataFrame(
            False, index=self.data.index, columns=numeric_cols
        )

        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) < 2:
                continue

            mean = cast(float, col_data.mean())
            std_val = cast(float, col_data.std())
            if not (std_val > 0):
                continue

            zscore = ((self.data[col] - mean) / std_val).abs()
            mask[col] = zscore > threshold

        self.outlier_masks["zscore"] = mask
        return mask

    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.

        Parameters:
            multiplier (float): IQR multiplier for boundary calculation

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        mask: pd.DataFrame = pd.DataFrame(
            False, index=self.data.index, columns=numeric_cols
        )

        for col in numeric_cols:
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue

            q1 = cast(float, col_data.quantile(0.25))
            q3 = cast(float, col_data.quantile(0.75))
            iqr = q3 - q1
            if not (iqr > 0):
                continue

            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            mask[col] = (self.data[col] < lower) | (self.data[col] > upper)

        self.outlier_masks["iqr"] = mask
        return mask

    def detect_outliers_isolation_forest(
        self, contamination: float = 0.1, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.

        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] == 0 or len(numeric_data) == 0:
            mask: pd.DataFrame = pd.DataFrame(
                {"is_outlier": False}, index=self.data.index
            )
            self.outlier_masks["isolation_forest"] = mask
            return mask

        numeric_data = numeric_data.fillna(0)

        iso = IsolationForest(
            contamination=contamination,  # type: ignore
            random_state=random_state,
        )
        predictions = iso.fit_predict(numeric_data)

        mask = pd.DataFrame({"is_outlier": predictions == -1}, index=self.data.index)
        self.outlier_masks["isolation_forest"] = mask
        return mask

    def detect_outliers_lof(
        self, n_neighbors: int = 20, contamination: float = 0.1
    ) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.

        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers

        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] == 0 or len(numeric_data) == 0:
            mask: pd.DataFrame = pd.DataFrame(
                {"is_outlier": False}, index=self.data.index
            )
            self.outlier_masks["lof"] = mask
            return mask

        numeric_data = numeric_data.fillna(0)
        n_neighbors = min(n_neighbors, max(1, len(numeric_data) - 1))

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,  # type: ignore
        )
        predictions = lof.fit_predict(numeric_data)

        mask = pd.DataFrame({"is_outlier": predictions == -1}, index=self.data.index)
        self.outlier_masks["lof"] = mask
        return mask

    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.

        Returns:
            DataFrame with outlier counts for each method
        """
        if not self.outlier_masks:
            return pd.DataFrame()

        summary: dict[str, dict[str, int]] = {}
        for method, mask in self.outlier_masks.items():
            total_cells = int(mask.sum().sum())
            rows_with = int(mask.any(axis=1).sum())
            summary[method] = {
                "total_outlier_cells": total_cells,
                "rows_with_outliers": rows_with,
            }

        return pd.DataFrame(summary)

    def remove_outliers(self, method: str = "iqr") -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Parameters:
            method (str): Detection method to use

        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if method not in self.outlier_masks:
            if method == "zscore":
                self.detect_outliers_zscore()
            elif method == "iqr":
                self.detect_outliers_iqr()
            elif method == "isolation_forest":
                self.detect_outliers_isolation_forest()
            elif method == "lof":
                self.detect_outliers_lof()
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

        mask = self.outlier_masks[method]
        rows_to_remove = mask.any(axis=1)

        cleaned: pd.DataFrame = self.data.loc[~rows_to_remove].copy()
        self.treated_datasets[f"removed_{method}"] = cleaned
        return cleaned

    def cap_outliers(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.

        Parameters:
            multiplier (float): IQR multiplier for bounds

        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        capped: pd.DataFrame = self.data.copy()
        numeric_cols = capped.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = capped[col].dropna()
            if len(col_data) == 0:
                continue

            q1 = cast(float, col_data.quantile(0.25))
            q3 = cast(float, col_data.quantile(0.75))
            iqr = q3 - q1
            if not (iqr > 0):
                continue

            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            capped[col] = capped[col].clip(lower=lower, upper=upper)

        self.treated_datasets["capped"] = capped
        return capped

    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.

        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        transformed: pd.DataFrame = self.data.copy()
        numeric_cols = transformed.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = transformed[col]
            min_val = col_data.min()

            if min_val <= 0:
                transformed[col] = np.log(col_data - min_val + 1)
            else:
                transformed[col] = np.log1p(col_data)

        self.treated_datasets["log_transform"] = transformed
        return transformed

    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.

        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        transformed: pd.DataFrame = self.data.copy()
        numeric_cols = transformed.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0 or len(transformed) == 0:
            self.treated_datasets["boxcox_transform"] = transformed
            return transformed

        pt = PowerTransformer(method="yeo-johnson")
        transformed[numeric_cols] = pt.fit_transform(transformed[numeric_cols])

        self.treated_datasets["boxcox_transform"] = transformed
        return transformed

    def evaluate_model_performance(
        self, target_col: str, dataset_name: str = "original"
    ) -> dict[str, float]:
        """
        Evaluate model performance with and without outliers.

        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)

        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == "original":
            dataset = self.original_data
        elif dataset_name in self.treated_datasets:
            dataset = self.treated_datasets[dataset_name]
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available: 'original', {list(self.treated_datasets.keys())}"
            )

        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        feature_cols = [
            c
            for c in dataset.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]

        if not feature_cols:
            return {"r2_score": 0.0, "r2_std": 0.0, "mean_cv_score": 0.0}

        df_clean = dataset[[target_col] + feature_cols].dropna()
        if len(df_clean) < 2:
            return {"r2_score": 0.0, "r2_std": 0.0, "mean_cv_score": 0.0}

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        n_splits = min(5, len(df_clean))
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=n_splits, scoring="r2")

        return {
            "r2_score": float(scores.mean()),
            "r2_std": float(scores.std()),
            "mean_cv_score": float(scores.mean()),
        }

    def visualize_outliers_boxplot(self, figsize: tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.

        Parameters:
            figsize (Tuple[int, int]): Figure size for plots

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        n_plots = min(len(numeric_cols), 6)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx in range(n_plots):
            col = numeric_cols[idx]
            ax = axes[idx]
            ax.boxplot(
                self.data[col].dropna(),
                patch_artist=True,
                boxprops={"facecolor": "lightblue", "alpha": 0.7},
                medianprops={"color": "red", "linewidth": 2},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 6},
            )

            # Highlight z-score outliers if available
            if (
                "zscore" in self.outlier_masks
                and col in self.outlier_masks["zscore"].columns
            ):
                outlier_vals = self.data.loc[self.outlier_masks["zscore"][col], col]
                if len(outlier_vals) > 0:
                    ax.scatter(
                        np.ones(len(outlier_vals)),
                        outlier_vals,
                        color="red",
                        s=50,
                        zorder=5,
                    )

            ax.set_title(f"Boxplot: {col}", fontweight="bold")
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)

        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Box Plots for Outlier Detection",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return fig

    def visualize_outliers_violin(self, figsize: tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.

        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        n_plots = len(numeric_cols)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx in range(n_plots):
            col = numeric_cols[idx]
            ax = axes[idx]
            parts = ax.violinplot(
                self.data[col].dropna(), showmeans=True, showmedians=True
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("lightblue")
                pc.set_alpha(0.7)
            ax.set_title(f"Violin Plot: {col}", fontweight="bold")
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)

        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Violin Plots for Outlier Detection",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return fig

    def visualize_outliers_scatter(
        self, x_col: str, y_col: str, figsize: tuple[int, int] = (15, 5)
    ):
        """
        Create scatter plots showing outliers detected by different methods.

        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Column(s) '{x_col}' or '{y_col}' not found")
            return None

        methods = list(self.outlier_masks.keys())
        n_plots = max(len(methods), 1)

        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes_list = [axes]
        else:
            axes_list = list(np.asarray(axes).flatten())

        for idx in range(n_plots):
            ax = axes_list[idx]
            if idx < len(methods):
                method = methods[idx]
                mask = self.outlier_masks[method]

                if x_col in mask.columns and y_col in mask.columns:
                    combined = mask[x_col] | mask[y_col]
                else:
                    combined = mask.any(axis=1)

                normal = ~combined
                ax.scatter(
                    self.data.loc[normal, x_col],
                    self.data.loc[normal, y_col],
                    alpha=0.5,
                    c="steelblue",
                    label="Normal",
                    s=30,
                )
                ax.scatter(
                    self.data.loc[combined, x_col],
                    self.data.loc[combined, y_col],
                    alpha=0.9,
                    c="red",
                    label="Outlier",
                    s=80,
                    edgecolors="black",
                )
                ax.set_title(method.replace("_", " ").title(), fontweight="bold")
                ax.legend()
            else:
                ax.scatter(self.data[x_col], self.data[y_col], alpha=0.5, c="steelblue")
                ax.set_title("No Detection Performed", fontweight="bold")

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(alpha=0.3)

        plt.suptitle(
            f"Outlier Detection: {x_col} vs {y_col}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return fig

    def visualize_3d_outliers(
        self,
        x_col: str,
        y_col: str,
        z_col: str,
        method: str = "isolation_forest",
        figsize: tuple[int, int] = (12, 8),
    ):
        """
        Create three-dimensional scatter plot showing multivariate outliers.

        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        from mpl_toolkits.mplot3d import Axes3D

        if (
            x_col not in self.data.columns
            or y_col not in self.data.columns
            or z_col not in self.data.columns
        ):
            print(f"Column(s) '{x_col}', '{y_col}', or '{z_col}' not found")
            return None

        if method not in self.outlier_masks:
            print(
                f"Outlier detection method '{method}' not found. Run detection first."
            )
            return None

        mask = self.outlier_masks[method]
        combined = mask.any(axis=1)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        normal = ~combined
        ax.scatter(
            self.data.loc[normal, x_col],
            self.data.loc[normal, y_col],
            self.data.loc[normal, z_col],
            alpha=0.5,
            c="steelblue",
            label="Normal",
            s=30,
        )
        ax.scatter(
            self.data.loc[combined, x_col],
            self.data.loc[combined, y_col],
            self.data.loc[combined, z_col],
            alpha=0.9,
            c="red",
            label="Outlier",
            s=100,
            edgecolors="black",
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(
            f"3D Outlier Visualization: {method.replace('_', ' ').title()}",
            fontweight="bold",
        )
        ax.legend()

        plt.tight_layout()
        return fig

    def visualize_treatment_comparison(
        self, column: str, figsize: tuple[int, int] = (15, 10)
    ):
        """
        Compare distributions before and after different treatments.

        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # Original data
        ax = axes[0]
        self.original_data[column].hist(
            bins=30, ax=ax, color="skyblue", edgecolor="black", alpha=0.7
        )
        ax.axvline(
            self.original_data[column].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        ax.set_title("Original Data", fontweight="bold")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(
                    bins=30, ax=ax, color="lightcoral", edgecolor="black", alpha=0.7
                )
                ax.axvline(
                    treated_data[column].mean(),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Mean",
                )
                ax.set_title(f"{name.replace('_', ' ').title()}", fontweight="bold")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Treatment Comparison: {column}", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        return fig

    def visualize_qq_plots(self, figsize: tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.

        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]

        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]

            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {col}", fontweight="bold")
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Q-Q Plots for Normality Assessment", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.

    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300

    # Create base data
    data = {
        "feature1": np.random.normal(100, 15, n_samples),
        "feature2": np.random.exponential(50, n_samples),
        "feature3": np.random.normal(500, 100, n_samples),
        "feature4": np.random.uniform(0, 100, n_samples),
        "target": np.random.normal(200, 30, n_samples),
    }

    df = pd.DataFrame(data)

    # Add synthetic target relationship
    df["target"] = (
        0.5 * df["feature1"] + 0.3 * df["feature2"] + np.random.normal(0, 10, n_samples)
    )

    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, "feature1"] = np.random.uniform(
        200, 300, len(outlier_indices)
    )

    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, "feature2"] = np.random.uniform(
        200, 400, len(outlier_indices)
    )

    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[: len(outlier_indices) // 2]
    high_outliers = outlier_indices[len(outlier_indices) // 2 :]
    df.loc[low_outliers, "feature3"] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, "feature3"] = np.random.uniform(
        1000, 1500, len(high_outliers)
    )

    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.

    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()

    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()

    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)

    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)

    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")

    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")

    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")

    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()

    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()

    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)

    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method="iqr")
    print(
        f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})"
    )

    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print("     Data capped at IQR bounds")

    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print("     Log transformation applied")

    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print("     Box-Cox/Yeo-Johnson transformation applied")
    print()

    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)

    datasets_to_evaluate = ["original", "removed_iqr", "capped", "log_transform"]
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance("target", dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(
                f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})"
            )
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()

    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)

    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig("task2_boxplots.png", dpi=300, bbox_inches="tight")
    print("  Saved: task2_boxplots.png")

    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig("task2_violin_plots.png", dpi=300, bbox_inches="tight")
    print("  Saved: task2_violin_plots.png")

    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter("feature1", "feature2")
    plt.savefig("task2_scatter_outliers.png", dpi=300, bbox_inches="tight")
    print("  Saved: task2_scatter_outliers.png")

    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers(
        "feature1", "feature2", "feature3", method="isolation_forest"
    )
    plt.savefig("task2_3d_outliers.png", dpi=300, bbox_inches="tight")
    print("  Saved: task2_3d_outliers.png")

    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison("feature1")
    plt.savefig("task2_treatment_comparison.png", dpi=300, bbox_inches="tight")
    print("  Saved: task2_treatment_comparison.png")

    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig("task2_qq_plots.png", dpi=300, bbox_inches="tight")
    print("  Saved: task2_qq_plots.png")
    print()

    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)

    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
