import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import os

def process_data(same_path, diff_path):
    """
    Xử lý dữ liệu từ các file CSV và tính toán các metrics.
    """
    df_same = pd.read_csv(same_path)
    df_diff = pd.read_csv(diff_path)

    df_same["Label"] = 1
    df_diff["Label"] = 0
    df = pd.concat([df_same, df_diff], ignore_index=True)

    thresholds = np.arange(0.05, 1.01, 0.01)
    results = []

    for thresh in thresholds:
        preds = (df["Similarity"] >= thresh).astype(int)
        true_labels = df["Label"]

        TP = np.sum((preds == 1) & (true_labels == 1))
        TN = np.sum((preds == 0) & (true_labels == 0))
        FP = np.sum((preds == 1) & (true_labels == 0))
        FN = np.sum((preds == 0) & (true_labels == 1))

        far = FP / (FP + TN) if (FP + TN) > 0 else 0
        frr = FN / (FN + TP) if (FN + TP) > 0 else 0
        err = (far + frr) / 2

        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, zero_division=0)

        results.append([thresh, acc, f1, far, frr, err])

    results_df = pd.DataFrame(results, columns=["Threshold", "Accuracy", "F1", "FAR", "FRR", "ERR"])

    # Tìm ERR (Equal Error Rate)
    err_idx = np.argmin(np.abs(results_df["FAR"] - results_df["FRR"]))
    err_threshold = results_df.iloc[err_idx]["Threshold"]
    err_value = (results_df.iloc[err_idx]["FAR"] + results_df.iloc[err_idx]["FRR"]) / 2

    # Tính ROC
    fpr, tpr, _ = roc_curve(df["Label"], df["Similarity"])
    roc_auc = auc(fpr, tpr)

    return results_df, fpr, tpr, roc_auc, err_threshold, err_value

def calculate_metrics_and_plots(same_path, diff_path):
    """
    Tính toán metrics và tạo biểu đồ Plotly.
    """
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(same_path) or not os.path.exists(diff_path):
        raise FileNotFoundError(f"File không tồn tại: {same_path} hoặc {diff_path}")

    # Xử lý dữ liệu
    results_df, fpr, tpr, roc_auc, err_threshold, err_value = process_data(same_path, diff_path)

    # Tạo các biểu đồ với Plotly
    charts = {}

    # Biểu đồ 1: Accuracy & F1
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=results_df["Threshold"],
        y=results_df["Accuracy"],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='blue', width=2),
        hovertemplate='Threshold: %{x:.2f}<br>Accuracy: %{y:.4f}'
    ))
    fig1.add_trace(go.Scatter(
        x=results_df["Threshold"],
        y=results_df["F1"],
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='green', width=2),
        hovertemplate='Threshold: %{x:.2f}<br>F1 Score: %{y:.4f}'
    ))
    fig1.update_layout(
        title='Accuracy và F1 Score',
        xaxis_title='Threshold',
        yaxis_title='Score',
        hovermode='closest',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    charts['accuracy_f1'] = fig1.to_html(full_html=False, include_plotlyjs='cdn')

    # Biểu đồ 2: FAR, FRR & ERR
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results_df["Threshold"],
        y=results_df["FAR"],
        mode='lines+markers',
        name='FAR',
        line=dict(color='red', width=2),
        hovertemplate='Threshold: %{x:.2f}<br>FAR: %{y:.4f}'
    ))
    fig2.add_trace(go.Scatter(
        x=results_df["Threshold"],
        y=results_df["FRR"],
        mode='lines+markers',
        name='FRR',
        line=dict(color='blue', width=2),
        hovertemplate='Threshold: %{x:.2f}<br>FRR: %{y:.4f}'
    ))
    # Add vertical line for ERR
    fig2.add_shape(
        type="line",
        x0=err_threshold,
        y0=0,
        x1=err_threshold,
        y1=1,
        line=dict(color="red", width=2, dash="dash"),
    )
    # Add annotation for ERR
    fig2.add_annotation(
        x=err_threshold,
        y=err_value + 0.05,
        text=f"ERR = {err_value:.4f} at {err_threshold:.4f}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-40
    )

    # Find optimal threshold (already calculated later in the code)
    optimal_threshold = results_df.loc[results_df["F1"].idxmax(), "Threshold"]
    optimal_idx = results_df["Threshold"].sub(optimal_threshold).abs().idxmin()
    optimal_far = results_df.iloc[optimal_idx]["FAR"]
    optimal_frr = results_df.iloc[optimal_idx]["FRR"]

    # Add vertical line for optimal threshold
    fig2.add_shape(
        type="line",
        x0=optimal_threshold,
        y0=0,
        x1=optimal_threshold,
        y1=1,
        line=dict(color="green", width=2, dash="dot"),
    )
    # Add annotation for optimal threshold
    fig2.add_annotation(
        x=optimal_threshold,
        y=0.8,
        text=f"Optimal: FAR={optimal_far:.4f}, FRR={optimal_frr:.4f} at {optimal_threshold:.4f}",
        showarrow=True,
        arrowhead=1,
        ax=-80,
        ay=-40
    )
    fig2.update_layout(
        title='FAR-FRR Curve',
        xaxis_title='Threshold',
        yaxis_title='Rate',
        hovermode='closest',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    charts['far_frr'] = fig2.to_html(full_html=False, include_plotlyjs='cdn')

    # Biểu đồ 3: ROC Curve
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2),
        hovertemplate='False Positive Rate: %{x:.4f}<br>True Positive Rate: %{y:.4f}'
    ))
    # Add diagonal line
    fig3.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='navy', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    fig3.update_layout(
        title='Receiver Operating Characteristic (ROC)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='closest',
        legend=dict(x=0.02, y=0.02),
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    charts['roc'] = fig3.to_html(full_html=False, include_plotlyjs='cdn')

    # Tìm ngưỡng tối ưu dựa trên F1 score cao nhất
    optimal_threshold = results_df.loc[results_df["F1"].idxmax(), "Threshold"]

    # Lấy các metrics tại ngưỡng tối ưu
    optimal_idx = results_df["Threshold"].sub(optimal_threshold).abs().idxmin()
    optimal_metrics = results_df.iloc[optimal_idx]

    # Tính toán các metrics để hiển thị
    metrics = {
        'err': err_value,
        'err_threshold': err_threshold,
        'auc': roc_auc,
        'accuracy': optimal_metrics["Accuracy"],
        'f1_score': optimal_metrics["F1"],
        'far': optimal_metrics["FAR"],
        'frr': optimal_metrics["FRR"],
        'optimal_threshold': optimal_threshold
    }

    return charts, metrics
