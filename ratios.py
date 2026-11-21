import numpy as np

def add_ratios(df):
    df = df.copy()

    def ratio(a, b):
        return (a / b.replace(0, np.nan)).fillna(0)

    df["commits_per_active_day"] = ratio(df["num_commits"], df["active_commit_days"])
    df["active_day_density"] = df["active_commit_days"] / (df["project_span_days"] + 1)
    df["lines_per_active_day"] = ratio(df["lines_added_total"], df["active_commit_days"])
    df["lines_per_commit"] = ratio(df["lines_added_total"], df["num_commits"])
    df["loc_per_file"] = ratio(df["lines_of_code"], df["num_source_files"])
    df["total_assets"] = (df["num_image_files"] + df["num_audio_files"]
                          + df["num_3d_model_files"] + df["num_font_files"])
    df["top_commit_share"] = ratio(df["max_lines_in_one_commit"], df["lines_added_total"])
    return df