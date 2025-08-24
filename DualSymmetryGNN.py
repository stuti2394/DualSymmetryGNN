# Optuna-enhanced DualSymmetryGNN full pipeline with full feature set, evaluation, plots, and activation optimization

# Imports
# Optuna-enhanced DualSymmetryGNN full pipeline with full feature set, evaluation, plots, and activation optimization

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score, confusion_matrix, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, TransformerConv, SAGEConv, GINConv, global_add_pool, global_max_pool
import optuna.visualization as vis

# -------------------------- DATA UTILS --------------------------
def parse_coordinates_string(coord_str):
    coord_str = str(coord_str).strip('[]')
    matches = re.findall(r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)', coord_str)
    return [(float(x), float(y)) for x, y in matches]

def load_polygon_dataset_from_excel(file_path):
    dataset = []
    df = pd.read_excel(file_path)
    for idx, row in df.iterrows():
        try:
            coord_str = row['coordinates']
            vertices = parse_coordinates_string(coord_str)
            if len(vertices) < 3:
                continue
            symmetry_lines = float(row['symmetry_lines'])
            edges = []
            n = len(vertices)
            for i in range(n):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % n])
                dist = np.linalg.norm(p1 - p2)
                edges.append((i, (i + 1) % n, dist))
            dataset.append({'label': symmetry_lines, 'vertices': vertices, 'edges': edges, 'num_points': len(vertices)})
        except:
            continue
    return dataset

def normalize_polygon(vertices):
    vertices = np.array(vertices)
    centroid = np.mean(vertices, axis=0)
    normalized = vertices - centroid
    max_dist = np.max(np.linalg.norm(normalized, axis=1))
    if max_dist > 0:
        normalized = normalized / max_dist
    return normalized

def compute_symmetry_descriptors(vertices):
    n = len(vertices)
    descriptors = []
    axis_x = 0.0
    for i in range(n):
        x_, y_ = vertices[i]
        dist_to_axis = abs(x_ - axis_x)
        mirror_x = 2 * axis_x - x_
        dists = np.linalg.norm(vertices - np.array([mirror_x, y_]), axis=1)
        min_dist = np.min(dists)
        descriptors.append([dist_to_axis, min_dist])
    return np.array(descriptors)

def compute_rotational_symmetry_features(vertices):
    n = len(vertices)
    center = np.mean(vertices, axis=0)
    relative_vertices = vertices - center
    angles = np.arctan2(relative_vertices[:, 1], relative_vertices[:, 0])
    radii = np.linalg.norm(relative_vertices, axis=1)
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_radii = radii[sorted_indices]
    features = []
    for k in range(2, min(n+1, 13)):
        if n % k == 0:
            angle_step = 2 * np.pi / k
            symmetry_score = 0
            for i in range(n):
                expected_angle = (sorted_angles[0] + (i % k) * angle_step) % (2 * np.pi)
                actual_angle = sorted_angles[i] % (2 * np.pi)
                angle_diff = min(abs(expected_angle - actual_angle), 2 * np.pi - abs(expected_angle - actual_angle))
                radius_group = i // k
                if radius_group < len(sorted_radii) // k:
                    expected_radius = sorted_radii[i % k]
                    actual_radius = sorted_radii[i]
                    radius_diff = abs(expected_radius - actual_radius)
                    symmetry_score += np.exp(-10 * (angle_diff + radius_diff))
            features.append(symmetry_score / n)
        else:
            features.append(0.0)
    while len(features) < 11:
        features.append(0.0)
    return np.array(features[:11])

def compute_feature_lines(vertices):
    n = len(vertices)
    feature_lines = []
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length > 0:
            vec = vec / length
        feature_lines.append(np.concatenate([vec, [length]]))
    return np.array(feature_lines)

def compute_vertex_angles(vertices):
    n = len(vertices)
    angles = []
    for i in range(n):
        p_prev = vertices[(i - 1) % n]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-9)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-9)
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)
    return np.array(angles).reshape(-1, 1)

def polygon_to_graph(polygon):
    vertices = normalize_polygon(polygon['vertices'])
    n = len(vertices)
    symmetry_desc = compute_symmetry_descriptors(vertices)
    feature_lines = compute_feature_lines(vertices)
    angles = compute_vertex_angles(vertices)
    degrees = np.zeros((n, 1))
    for e in polygon['edges']:
        degrees[e[0]] += 1
        degrees[e[1]] += 1
    global_features = np.tile(compute_rotational_symmetry_features(vertices), (n, 1))
    vertex_features = np.hstack([vertices, symmetry_desc, feature_lines, degrees, angles, global_features])
    mean = vertex_features.mean(axis=0, keepdims=True)
    std = vertex_features.std(axis=0, keepdims=True) + 1e-9
    x = (vertex_features - mean) / std
    edge_index = []
    edge_attr = []
    for e in polygon['edges']:
        edge_index.append([e[0], e[1]])
        edge_index.append([e[1], e[0]])
        edge_attr.append([e[2]])
        edge_attr.append([e[2]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index,
                edge_attr=edge_attr, y=torch.tensor([polygon['label']], dtype=torch.float),
                num_points=torch.tensor([polygon['num_points']], dtype=torch.long))

def load_and_process_graphs(file):
    raw_polygons = load_polygon_dataset_from_excel(file)
    graphs = [polygon_to_graph(poly) for poly in raw_polygons]
    print(f"✅ Loaded {len(graphs)} graphs")
    if len(graphs) < 2:
        raise ValueError("Not enough data for training. Please check the Excel file or parsing logic.")
    return graphs


# ---------- Model Architecture ----------


def get_activation_fn(name):
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'prelu': nn.PReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softplus': nn.Softplus(),
        'selu': nn.SELU(),
        'mish': nn.Mish()
    }[name]

def get_conv(conv_type, in_channels, out_channels, edge_dim, heads=1):
    if conv_type == "GATv2":
        return GATv2Conv(in_channels, out_channels, edge_dim=edge_dim, heads=heads)
    elif conv_type == "Transformer":
        return TransformerConv(in_channels, out_channels, edge_dim=edge_dim, heads=heads)
    elif conv_type == "SAGE":
        return SAGEConv(in_channels, out_channels)
    elif conv_type == "GIN":
        nn1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        return GINConv(nn1)
    elif conv_type == "GCN":
        return GCNConv(in_channels, out_channels)
    else:
        raise ValueError("Unsupported conv type")

class DualSymmetryGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3, use_residual=True, conv_type="GATv2", activation="relu"):
        super().__init__()
        self.conv_type = conv_type
        self.use_residual = use_residual
        self.act = get_activation_fn(activation)

        factor = 4 if conv_type in ["GATv2", "Transformer"] else 1

        self.conv1 = get_conv(conv_type, in_dim, hidden_dim, edge_dim=1, heads=4)
        self.norm1 = nn.LayerNorm(hidden_dim * factor)

        self.conv2 = get_conv(conv_type, hidden_dim * factor, hidden_dim, edge_dim=1, heads=4)
        self.norm2 = nn.LayerNorm(hidden_dim * factor)

        self.conv3 = get_conv(conv_type, hidden_dim * factor, hidden_dim, edge_dim=1)

        if use_residual:
            self.res1 = nn.Linear(hidden_dim * factor, hidden_dim * factor)
            self.res2 = nn.Linear(hidden_dim * factor, hidden_dim)

        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), self.act, nn.Dropout(dropout))

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), self.act, nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), self.act, nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 2))

    def forward(self, data, task='both'):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.conv_type in ["GIN", "SAGE", "GCN"]:
            edge_attr = None

        x1 = self.act(self.conv1(x, edge_index, edge_attr))
        x1 = self.norm1(x1)

        x2 = self.act(self.conv2(x1, edge_index, edge_attr))
        x2 = self.norm2(x2)
        if self.use_residual:
            x2 = x2 + self.res1(x1)

        x3 = self.act(self.conv3(x2, edge_index, edge_attr))
        if self.use_residual:
            x3 = x3 + self.res2(x2)

        x_add = global_add_pool(x3, data.batch)
        x_max = global_max_pool(x3, data.batch)
        x = torch.cat([x_add, x_max], dim=1)

        shared = self.shared_fc(x)

        if task == 'regression':
            return self.regression_head(shared)
        elif task == 'binary':
            return self.binary_head(shared)
        else:
            return self.regression_head(shared), self.binary_head(shared)


# ---------- Training ----------

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_total = 0
    reg_crit = nn.MSELoss()
    bin_crit = nn.CrossEntropyLoss()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        reg_pred, bin_pred = model(batch)
        reg_loss = reg_crit(reg_pred.squeeze(), batch.y.squeeze())
        binary_targets = (batch.y.squeeze() > 0).long()
        bin_loss = bin_crit(bin_pred, binary_targets)
        loss = reg_loss + bin_loss
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(loader)


# ---------- Evaluation + Plotting ----------

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_bin_pred, sizes = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reg_out, bin_out = model(batch)
            y_true += batch.y.cpu().numpy().tolist()
            y_pred += reg_out.squeeze().cpu().numpy().tolist()
            y_bin_pred += bin_out.argmax(dim=1).cpu().numpy().tolist()
            sizes += batch.num_points.cpu().numpy().tolist()

    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 0, None)
    y_bin_true = (y_true > 0).astype(int)
    y_bin_pred = np.array(y_bin_pred)

    result = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Exact': accuracy_score(y_true, np.round(y_pred)),
        'BinAcc': accuracy_score(y_bin_true, y_bin_pred),
        'BinF1': f1_score(y_bin_true, y_bin_pred),
        'ConfMat': confusion_matrix(y_bin_true, y_bin_pred)
    }

    df = pd.DataFrame({
        'size': sizes,
        'true': y_true,
        'pred': np.round(y_pred),
        'bin_true': y_bin_true,
        'bin_pred': y_bin_pred
    })

    grouped = df.groupby('size')
    reg_acc = grouped.apply(lambda x: accuracy_score(x['true'], x['pred']))
    bin_acc = grouped.apply(lambda x: accuracy_score(x['bin_true'], x['bin_pred']))

    plt.figure(figsize=(10, 5))
    plt.bar(reg_acc.index - 0.2, reg_acc.values, width=0.4, label='Regression')
    plt.bar(bin_acc.index + 0.2, bin_acc.values, width=0.4, label='Binary')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Polygon Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    return result


# ---------- Optuna Objective ----------

def objective(trial, graphs):
    hidden = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    conv_type = trial.suggest_categorical("conv", ["GATv2", "Transformer", "SAGE", "GIN"])
    use_res = trial.suggest_categorical("residual", [True, False])
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "gelu", "prelu", "sigmoid", "tanh", "softplus"])

    y = [g.y.item() for g in graphs]
    y_bin = [1 if i > 0 else 0 for i in y]
    if len(set(y_bin)) < 2:
        raise ValueError("Insufficient data variety for stratified validation split")

    from sklearn.model_selection import train_test_split
    train, val = train_test_split(graphs, test_size=0.2, stratify=y_bin, random_state=42)

    model = DualSymmetryGNN(
        in_dim=graphs[0].x.size(1),
        hidden_dim=hidden,
        dropout=dropout,
        conv_type=conv_type,
        use_residual=use_res,
        activation=activation
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train, batch_size=32)
    val_loader = DataLoader(val, batch_size=32)

    for _ in range(20):
        train_epoch(model, train_loader, optimizer, device)

    val_result = evaluate(model, val_loader, device)
    return val_result['RMSE']

# ---------- Main ----------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full training data (no split)
    print("Loading full training data...")
    full_train_graphs = load_and_process_graphs("/kaggle/input/new-input/polygons_new.xlsx")

    # Run Optuna
    print("Running Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, full_train_graphs), n_trials=30)

    print("Best hyperparameters found:")
    print(study.best_trial.params)

    # Visualizations
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()

    # Retrain model on entire training dataset using best parameters
    print("Training final model on full dataset...")
    best_params = study.best_trial.params
    model = DualSymmetryGNN(
        in_dim=full_train_graphs[0].x.size(1),
        hidden_dim=best_params['hidden_dim'],
        dropout=best_params['dropout'],
        conv_type=best_params['conv'],
        use_residual=best_params['residual'],
        activation=best_params['activation']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(full_train_graphs, batch_size=32, shuffle=True)

    losses = []
    for epoch in range(1, 31):
        loss = train_epoch(model, train_loader, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f}")

    # Plot training loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()

    # Load and evaluate on new dataset
    print("Loading new test set for evaluation...")
    new_graphs = load_and_process_graphs('/kaggle/input/new-input/new.xlsx')
    new_loader = DataLoader(new_graphs, batch_size=32)
    
    print(" Evaluating on new dataset...")
    new_result = evaluate(model, new_loader, device)
    print("Evaluation results on NEW dataset:")
    for k, v in new_result.items():
        print(f"{k}: {v}")

    # Save model
    print("Saving final model...")
    torch.save(model.state_dict(), "best_model.pt")
