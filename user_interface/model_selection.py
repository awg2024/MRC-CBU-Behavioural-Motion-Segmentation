from pathlib import Path
import pandas as pd
import numpy as np
import joblib  # for loading pickle models
import cv2
from sklearn.preprocessing import StandardScaler
import umap
import re 
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from hmmlearn.hmm import GaussianHMM
from scipy.stats import mode
import os 


# HARD CODED VALUES 

print("Behavioural motion classification script started. Please note GTK & SFO will not be classified.")

PARTICIPANT = ["PT_21"] #PT_05, PT_07
DAY = ["Day1"] # Day3, day5

# Classifier paths
CLASSIFIERS = {
    "Pegs": Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\pegs_behavioural_classifier.pkl"),
    "Jenga": Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\jenga_behavioural_classifier.pkl"),
    "Tapes": Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\tapes_behavioural_classifier.pkl"),
    "Eggs": Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\eggs_behavioural_classifier.pkl"),
    "Grasp": Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\grasp_behavioural_classifier.pkl"),
    "Coins": Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\coins_behavioural_classifier.pkl"),
}

BASE_PATH = Path(r"\\cbsu\data\Group\Plasticity\Projects\Null_space_markerless_tracking\Video_for_test")

# all trials that the model will analyse. 
TRIALS = ["Coins", "Eggs_First", "Eggs_Second", "Grasp", "Jenga", "Jenga_Standing",
          "Pegs", "Pegs_Standing", "Tapes", "Tapes_Standing"]

TRIAL_NOS = [1, 2, 3, 4, 5]

TASK_FALLBACKS = {
    "pegs_standing": "pegs",
    "gtk": "pegs",          
    "sfo": "pegs",          
    # Add any other fallback rules here
}

# helper functions 
def get_video_fps(video_path, default_fps=60):
    cap = cv2.VideoCapture(str(video_path))
    return cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else default_fps

def compute_velocity(pos, fps):
    dt = 1 / fps
    return np.vstack(([np.zeros(pos.shape[1])], np.diff(pos, axis=0) / dt))

def compute_aperture(df, joints):
    distances = {}
    for i, j1 in enumerate(joints):
        for j2 in joints[i+1:]:
            try:
                p1 = df[[f"{j1}_x", f"{j1}_y", f"{j1}_z"]].to_numpy()
                p2 = df[[f"{j2}_x", f"{j2}_y", f"{j2}_z"]].to_numpy()
    
                # vectorised. 
                dist = np.linalg.norm(p1 - p2, axis=1)

                distances[f"{j1}_to_{j2}"] = dist
            except KeyError:
                distances[f"{j1}_to_{j2}"] = np.zeros(len(df))
    return pd.DataFrame(distances)

def extract_joint_names(df):
    return sorted(set(col.rsplit('_', 1)[0] for col in df.columns if col.endswith('_x')))

def create_sliding_windows(features_df, window_size_sec=1.5, stride_sec=0.5, fps=60):
    window_size = int(window_size_sec * fps)
    stride = int(stride_sec * fps)
    windows, starts = [], []

    feature_cols = features_df.drop(columns=["frame", "time"]).columns

    for start in range(0, len(features_df) - window_size + 1, stride):
        end = start + window_size
        window = features_df.iloc[start:end]
        stats = window[feature_cols].agg(['mean', 'std', 'min', 'max']).values.flatten()
        windows.append(stats)
        starts.append(features_df.iloc[start]['frame'])

    return np.array(windows), np.array(starts)


# ---------- STEP 1: FEATURE EXTRACTION ----------
def feature_extraction(participant, day, task):
    print(f"\n📌 Starting Feature Extraction for {participant} {day} ({task})")

    base_path = BASE_PATH / "unprocessed" / f"{participant}_{day}"
    processed_path = BASE_PATH / "processed" / f"{participant}_{day}"
    processed_path.mkdir(exist_ok=True)

    trials = [t for t in TRIALS if task.lower() in t.lower() or task == ""]

    for trial in trials:
        for t_no in TRIAL_NOS:
            trial_name = f"{trial}_{t_no}"
            trial_path = base_path / trial_name / "all_3d" / "2001-01-01"
            pose_file = trial_path / "pose-3d" / f"{trial_name}.csv"
            angle_file = trial_path / "angles" / f"{trial_name}.csv"
            video_file = trial_path / "videos-raw" / f"{trial_name}-camA.mp4"
            output_csv = processed_path / f"{trial_name}_features.csv"

            if not (pose_file.exists() and video_file.exists()):
                print(f"❌ Skipping {trial_name}: missing files.")
                continue

            try:
                pose_df = pd.read_csv(pose_file)
                pose_df.columns = pose_df.columns.str.strip()

                if angle_file.exists():
                    angle_df = pd.read_csv(angle_file)
                    angle_df.columns = angle_df.columns.str.strip()
                    if 'fnum' in angle_df.columns:
                        pose_df['frame'] = np.arange(len(pose_df))
                        angle_df = angle_df.rename(columns={'fnum': 'frame'})
                        pose_df = pose_df.merge(angle_df, on='frame', how='left')
                    else:
                        pose_df = pd.concat([pose_df.reset_index(drop=True), angle_df.reset_index(drop=True)], axis=1)

                joints = extract_joint_names(pose_df)
                fps = get_video_fps(video_file)
                time = np.arange(len(pose_df)) / fps

                features = pd.DataFrame({'frame': np.arange(len(pose_df)), 'time': time})

                for joint in joints:
                    try:
                        pos = pose_df[[f"{joint}_x", f"{joint}_y", f"{joint}_z"]].values
                        vel = compute_velocity(pos, fps)
                        features[f"{joint}_vel"] = np.linalg.norm(vel, axis=1)
                        accel = np.vstack(([np.zeros(3)], np.diff(vel, axis=0) * fps))
                        features[f"{joint}_accel"] = np.linalg.norm(accel, axis=1)
                        delta_vel = np.vstack(([np.zeros(3)], np.diff(vel, axis=0)))
                        features[f"{joint}_delta_vel"] = np.linalg.norm(delta_vel, axis=1)
                        delta_accel = np.vstack(([np.zeros(3)], np.diff(accel, axis=0)))
                        features[f"{joint}_delta_accel"] = np.linalg.norm(delta_accel, axis=1)
                    except Exception:
                        features[f"{joint}_vel"] = np.zeros(len(pose_df))
                        features[f"{joint}_accel"] = np.zeros(len(pose_df))
                        features[f"{joint}_delta_vel"] = np.zeros(len(pose_df))
                        features[f"{joint}_delta_accel"] = np.zeros(len(pose_df))

                aperture_df = compute_aperture(pose_df, joints)
                features = pd.concat([features, aperture_df], axis=1)

                window_matrix, window_starts = create_sliding_windows(features, fps=fps)

                features.to_csv(output_csv, index=False)
                np.save(output_csv.with_suffix(".npy"), window_matrix)
                np.save(output_csv.with_suffix(".starts.npy"), window_starts)

                print(f"✅ {trial_name} processed: shape {window_matrix.shape}")

            except Exception as e:
                print(f"❌ Error processing {trial_name}: {e}")




# ---------- STEP 2: CLUSTERING ----------
def clustering(participant, day, task):
    print(f"\n🔍 Starting Clustering for {participant} {day} ({task})")



    PROCESSED_BASE_PATH = BASE_PATH / f"{participant}" / f"{participant}_{day}" / f"{task}" / "all_3d" / "2001-01-01"
    PROCESSED_BASE_PATH.mkdir(exist_ok=True)


    PROCESSED_PATH = PROCESSED_BASE_PATH / "behavioural_classification"
    PROCESSED_PATH.mkdir(exist_ok=True)

    FEATURE_PROCESSED_PATH = PROCESSED_PATH / "feature_output"
    FEATURE_PROCESSED_PATH.mkdir(exist_ok=True)


    CLUSTER_PROCESSED_PATH = PROCESSED_PATH / "clustering_output"
    CLUSTER_PROCESSED_PATH.mkdir(exist_ok=True)


    filtered_trials = [t for t in TRIALS if re.search(rf"\b{task}\b", t, re.IGNORECASE)] if task else TRIALS

    all_X, all_labels, all_frames, all_times, all_dfs = [], [], [], [], []
    feature_lengths = []

    # First pass: get max feature length
    for trial in filtered_trials:
        for t_no in TRIAL_NOS:
            npy_path = FEATURE_PROCESSED_PATH / f"{trial}_{t_no}_features.npy"
            starts_path = FEATURE_PROCESSED_PATH / f"{trial}_{t_no}_features.starts.npy"
            csv_path = FEATURE_PROCESSED_PATH / f"{trial}_{t_no}_features.csv"
            if npy_path.exists() and starts_path.exists() and csv_path.exists():
                X = np.load(npy_path)
                feature_lengths.append(X.shape[1])

    if not feature_lengths:
        raise RuntimeError(f"No feature files found for task '{task}' in {participant}_{day}.")
    max_features = max(feature_lengths)

    # Second pass: load and pad
    for trial in filtered_trials:
        for t_no in TRIAL_NOS:
            npy_path = FEATURE_PROCESSED_PATH / f"{trial}_{t_no}_features.npy"
            starts_path = FEATURE_PROCESSED_PATH / f"{trial}_{t_no}_features.starts.npy"
            csv_path = FEATURE_PROCESSED_PATH / f"{trial}_{t_no}_features.csv"
            if not (npy_path.exists() and starts_path.exists() and csv_path.exists()):
                continue

            X = np.load(npy_path)
            starts = np.load(starts_path)
            df = pd.read_csv(csv_path)

            if len(starts) != X.shape[0]:
                print(f"⚠️ Skipping {trial}_{t_no}: mismatch in starts vs windows.")
                continue

            if X.shape[1] < max_features:
                X = np.hstack([X, np.zeros((X.shape[0], max_features - X.shape[1]))])

            times = df.loc[df['frame'].isin(starts), 'time'].values

            all_X.append(X)
            all_frames.extend(starts)
            all_times.extend(times)
            all_labels.extend([trial] * len(starts))
            all_dfs.append(df[df['frame'].isin(starts)].reset_index(drop=True))

    combined_X = np.vstack(all_X)
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_X)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=28, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=25, cluster_selection_epsilon=0.01, prediction_data=True)
    labels = clusterer.fit_predict(X_umap)

    result_df = pd.DataFrame({
        "trial": all_labels,
        "time": all_times,
        "frame": all_frames,
        "UMAP_1": X_umap[:, 0],
        "UMAP_2": X_umap[:, 1],
        "cluster": labels
    })

    result_df = pd.concat([result_df.reset_index(drop=True), combined_df.reset_index(drop=True)], axis=1)

    # \\cbsu\data\Group\Plasticity\Projects\Null_space_markerless_tracking\Video_for_test\PT_14\PT_14_Day3\Coins_1\all_3d\2001-01-01


    output_csv = CLUSTER_PROCESSED_PATH / "cluster_output.csv"

    result_df.to_csv(output_csv, index=False)

    print(f"✅ Clustering results saved: {output_csv}")
    return result_df



# ---------- STEP 3: HMM ----------
def hmm_runner(participant, day, task, clustering_df):

    PROCESSED_BASE_PATH = BASE_PATH / f"{participant}" / f"{participant}_{day}" / f"{task}" / "all_3d" / "2001-01-01"
    
    HMM_PROCESSED_PATH = PROCESSED_BASE_PATH / "HMM_output"
    HMM_PROCESSED_PATH.mkdir(exist_ok=True)


    print(f"\n Running HMM for {participant} {day} ({task})")


    # --- One-hot encode cluster column ---
    cluster_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cluster_1hot = cluster_encoder.fit_transform(clustering_df[['cluster']])

    # --- Scale kinematic features ---
    kinematic_cols = [col for col in clustering_df.columns if col not in
                      ['trial', 'time', 'frame', 'cluster', 'UMAP_1', 'UMAP_2']]
    kinematic_data = clustering_df[kinematic_cols].fillna(0)
    kinematic_scaler = StandardScaler()
    kinematic_scaled = kinematic_scaler.fit_transform(kinematic_data)

    # --- Combine features for HMM ---
    hmm_input = np.hstack([kinematic_scaled, cluster_1hot])

    # --- BIC computation ---
    def compute_bic(model, X):
        n_components = model.n_components
        n_features = X.shape[1]
        trans_params = n_components * (n_components - 1)
        start_params = n_components - 1
        mean_params = n_components * n_features
        cov_params = n_components * n_features
        n_params = trans_params + start_params + mean_params + cov_params
        log_likelihood = model.score(X)
        return -2 * log_likelihood + n_params * np.log(X.shape[0])

    # --- Find best HMM ---
    lowest_bic = np.inf
    best_model = None
    for n in range(3, 6):
        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=200, verbose=True, random_state=42)
        model.fit(hmm_input)
        bic = compute_bic(model, hmm_input)
        print(f" States: {n} -> BIC: {bic:.2f}")
        if bic < lowest_bic:
            lowest_bic = bic
            best_model = model

    if best_model is None:
        raise RuntimeError("❌ HMM training failed.")

    # --- Predict states ---
    clustering_df['HMM_state'] = best_model.predict(hmm_input)
    clustering_df['HMM_state_smooth'] = (
        clustering_df['HMM_state']
        .rolling(window=3, center=True)
        .apply(lambda x: mode(x, keepdims=True)[0][0])
        .fillna(method='bfill')
        .fillna(method='ffill')
        .astype(int)
    )

    # --- SAVE RESULTS ---
    df_path = os.path.join(HMM_PROCESSED_PATH, f"HMM_{participant}_{day}_{task}.csv")
    model_path = os.path.join(HMM_PROCESSED_PATH, f"HMM_model_{participant}_{day}_{task}.joblib")

    clustering_df.to_csv(df_path, index=False)
    joblib.dump({
        "model": best_model,
        "scaler": kinematic_scaler,
        "encoder": cluster_encoder,
        "bic": lowest_bic
    }, model_path)

    print(f"💾 Saved HMM DataFrame -> {df_path}")
    print(f"💾 Saved HMM Model -> {model_path}")

    return clustering_df


# ---------- STEP 4: MODEL RUNNER ----------
def model_runner(participant, day, task, df, feature_cols):
    """Runs behavioural classifier."""
    print(f"\n🏷️ Running Behavioural Classifier for {task}")

    PROCESSED_BASE_PATH = BASE_PATH / f"{participant}" / f"{participant}_{day}" / f"{task}" / "all_3d" / "2001-01-01"

    MODEL_BASE_PATH = Path(r"\\cbsu\data\Group\Plasticity\Projects\Null_space_markerless_tracking\Behavioural_Classification") / f"{task.lower()}_behavioural_classifier.pkl"
    
    CLASSIFICATION_PATH = PROCESSED_BASE_PATH / "behavioural_classification_output"
    CLASSIFICATION_PATH.mkdir(exist_ok=True)


    if MODEL_BASE_PATH.exists():
        clf, feature_cols = joblib.load(MODEL_BASE_PATH)
        print("✅ Loaded existing classifier.")
    else:
        raise FileNotFoundError(f"No classifier found for task '{task}' at {MODEL_BASE_PATH}")

    # Keep only relevant features
    X = df[feature_cols].fillna(0)
    y_pred = clf.predict(X)
    df['predicted_label'] = y_pred

    print(f"✅ Predictions complete for {len(df)} rows.")

    # Save predictions
    output_file = CLASSIFICATION_PATH / f"{task}_predictions.csv"
    df[['frame', 'predicted_label']].to_csv(output_file, index=False)
    print(f"💾 Predictions saved to {output_file}")

    return df


# ---------- MAIN RUNNER ----------
def main_runner():
    print(f"🚀 Running pipeline for {PARTICIPANT} {DAY}")
    feature_extraction(PARTICIPANT, DAY, "")  # "" → all tasks

    for task in sorted(set(t.split("_")[0] for t in TRIALS)):
        try:

            clustering_df = clustering(PARTICIPANT, DAY, task)
            hmm_df = hmm_runner(PARTICIPANT, DAY, task, clustering_df)

            # You need to supply the actual feature column names for your classifier
            feature_cols = [col for col in hmm_df.columns if col not in ['trial', 'time', 'frame', 'cluster', 'UMAP_1', 'UMAP_2', 'HMM_state', 'HMM_state_smooth']]
            model_runner(PARTICIPANT, DAY, task, hmm_df, feature_cols)

        except RuntimeError as e:
            print(f"⚠️ Skipping task '{task}': {e}")
        except Exception as e:
            print(f"❌ Error on task '{task}': {e}")


if __name__ == "__main__":
    main_runner()
