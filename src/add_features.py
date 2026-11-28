import numpy as np

def add_features(df):
    df = df.copy()
    c = df["num_commits"].clip(lower=1)
    d = df["active_commit_days"].clip(lower=1)
    df["commits_per_day"] = c / d
    df["loc_per_commit"] = df["lines_of_code"] / c
    df["added_per_day"] = df["lines_added_total"] / d
    df["span_per_active_day"] = df["project_span_days"] / d
    df["log_loc"] = np.log1p(df["lines_of_code"])
    df["log_added"] = np.log1p(df["lines_added_total"])
    df["files_per_dir"] = df["num_source_files"] / df["num_directories"].clip(lower=1)
    df["asset_files"] = (df["num_image_files"] + df["num_audio_files"]
                         + df["num_3d_model_files"] + df["num_font_files"])
    df["test_ratio"] = df["num_test_files"] / df["num_source_files"].clip(lower=1)
    df["readme_effort"] = (df["readme_word_count"] + df["description_word_count"])
    return df.replace([np.inf, -np.inf], np.nan)