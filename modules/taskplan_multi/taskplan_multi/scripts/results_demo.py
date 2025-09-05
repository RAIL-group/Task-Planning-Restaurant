import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import re
import random
import numpy as np

DF_COLUMNS = ['seq', 'num', 'active', 'time', 'help', 'cost']

def natural_sort_key(s):
    """
    This function provides a key for sorting strings that contain numbers
    in a way that '2' comes before '10'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_tasks_comparison_cost(combined_df, smooth=True, method="ma", window=7, poly=2):
    def smooth_series(y, method="ma", window=7, poly=2):
        y = pd.Series(y).astype(float)
        if method == "ma":
            # Centered moving average; works with no extra deps
            return y.rolling(window, center=True, min_periods=1).mean().to_numpy()
        elif method == "savgol":
            # Requires scipy
            try:
                from scipy.signal import savgol_filter
                w = window + (1 - window % 2)  # ensure odd
                w = max(w, poly + 3)
                return savgol_filter(y.to_numpy(), w, poly)
            except Exception:
                return y.rolling(window, center=True, min_periods=1).mean().to_numpy()
        elif method == "lowess":
            # Requires statsmodels
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x = np.arange(len(y))
                frac = max(0.1, min(0.9, window / max(5, len(y))))
                return lowess(y, x, frac=frac, return_sorted=False)
            except Exception:
                return y.rolling(window, center=True, min_periods=1).mean().to_numpy()
        else:
            return y.to_numpy()

    color_map = {
        'No-Prep Myopic': 'cyan',
        'No-Prep Selfish A.P.': 'orangered',
        'No-Prep Proactive A.P': 'gold',
        'Prep Myopic': 'royalblue',
        'Prep Selfish A.P.': 'violet',
        'Prep Proactive A.P': 'pink',
    }
    marker_map = {
        'No-Prep Myopic': 'o',
        'No-Prep Selfish A.P.': 'd',
        'No-Prep Proactive A.P': 's',
        'Prep Myopic': 'o',
        'Prep Selfish A.P.': 'd',
        'Prep Proactive A.P': 's',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = combined_df['label'].unique()
    for label in labels:
        subset_df = combined_df[combined_df['label'] == label]
        subset_df = subset_df.sort_values(by='num', key=lambda x: x.map(natural_sort_key))

        x = subset_df['num']
        y = subset_df['avg_cost'].to_numpy()
        y_sm = smooth_series(y, method=method, window=window, poly=poly) if smooth else y

        # (1) Plot original (faint) for honesty
        ax.plot(
            x, y,
            marker=marker_map.get(label, '.'),
            color=color_map.get(label, 'black'),
            alpha=0.25, linewidth=1, label=None
        )

        # (2) Plot smoothed (bold) for readability
        ax.plot(
            x, y_sm,
            marker=None,  # keep line clean
            color=color_map.get(label, 'black'),
            linewidth=2.25,
            label=label
        )

        # Keep your original uncertainty band (unsmoothed) to avoid misleading CIs
        ax.fill_between(
            x,
            subset_df['avg_cost'] - subset_df['err_cost'],
            subset_df['avg_cost'] + subset_df['err_cost'],
            color=color_map.get(label, 'black'),
            alpha=0.1
        )

    ax.set_title('Average Cost Per Task', fontsize=10)
    ax.set_xlabel('Task Number', fontsize=10)
    ax.set_ylabel('Average Cost of Task Number')
    ax.legend(title='Planners', fontsize=10)
    ax.margins(x=0)
    plt.tight_layout()

    save_file = '/data/figs/compare-average-cost-new.png'
    plt.savefig(save_file, dpi=1200, bbox_inches='tight')


def get_df_group_by_seq(files):
    dfs = []
    for file_path in files:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, sep="\s*\|\s*", engine='python', header=None)
        df = df.dropna(axis=1, how='all')
        df.columns = DF_COLUMNS

        # Clean up the data by stripping whitespace and removing unnecessary characters
        df['seq'] = df['seq'].str.strip().str.split(':').str[1].str.strip()
        df['num'] = df['num'].str.strip().str.split(':').str[1].str.strip()
        df['cost'] = df['cost'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['fail'] = df['help'].str.strip().str.split(':').str[1].str.strip().astype(int)

        dfs.append(df)

    # Calculate the average cost group by 'task seq'

    merged_df = pd.concat(dfs, ignore_index=True)
    total_per_seq = df.groupby("num")["cost"].mean()
    average_cost = total_per_seq.mean()
    print("Average cost across sequences:", average_cost)
    return total_per_seq


def get_df_group_by_task(files):
    dfs = []
    for file_path in files:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, sep="\s*\|\s*", engine='python', header=None)
        df = df.dropna(axis=1, how='all')
        df.columns = DF_COLUMNS

        # Clean up the data by stripping whitespace and removing unnecessary characters
        df['seq'] = df['seq'].str.strip().str.split(':').str[1].str.strip()
        df['num'] = df['num'].str.strip().str.split(':').str[1].str.strip()
        df['cost'] = df['cost'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['help'] = df['help'].str.strip().str.split(':').str[1].str.strip().astype(int)
        df['time'] = df['time'].str.strip().str.split(':').str[1].str.strip().astype(float)

        dfs.append(df)

    # Calculate the average cost group by 'task seq'

    merged_df = pd.concat(dfs, ignore_index=True)
    result = merged_df.groupby("num").agg(
        avg_cost=("cost", "mean"),
        err_cost=("cost", "sem"),
        avg_help=("help", "mean"),
        err_help=("help", "sem"),
        avg_pt=("time", "mean"),
        err_pt=("time", "sem"),
        ).reset_index()
    result = result.sort_values(by='num', key=lambda x: x.map(natural_sort_key))
    return result
    


def normalize_desc(desc):
    # Normalize the format of the description to remove digits
    x = re.sub(r"\('(\w+?)\d*', '(\w+?)\d*'\)", r"('\1', '\2')", desc)
    if 'clean' in x:
        x = 'cleaning_tasks'
    if 'place' in x or 'clear' in x:
        x = 'organizing_tasks'
    return x

def compare(args):
    root = args.save_dir
    files_mp_np = list()
    files_ap_np_self = list()
    files_ap_np_other = list()
    files_ap_np_comb = list()
    files_mp_prep = list()
    files_ap_prep_self = list()
    files_ap_prep_other = list()
    files_ap_prep_comb = list()
    # for path, _, files in os.walk(root):
    #     for name in files:
    #         if 'np_myopic' in name:
    #             files_mp_np.append(os.path.join(path, name))
    #         elif 'np_ap_self' in name:
    #             files_ap_np_self.append(os.path.join(path, name))
    #         elif 'np_ap_other' in name:
    #             files_ap_np_other.append(os.path.join(path, name))
    #         elif 'np_ap_joint' in name:
    #             files_ap_np_comb.append(os.path.join(path, name))
    #         if 'prep_myopic' in name:
    #             files_mp_prep.append(os.path.join(path, name))
    #         elif 'prep_ap_self' in name:
    #             files_ap_prep_self.append(os.path.join(path, name))
    #         elif 'prep_ap_other' in name:
    #             files_ap_prep_other.append(os.path.join(path, name))
    #         elif 'prep_ap_joint' in name:
    #             files_ap_prep_comb.append(os.path.join(path, name))
    for path, _, files in os.walk(root):
        for name in files:
            if '_myopic' in name:
                files_mp_np.append(os.path.join(path, name))
            elif '_ap_self' in name:
                files_ap_np_self.append(os.path.join(path, name))
            # elif 'np_ap_other' in name:
            #     files_ap_np_other.append(os.path.join(path, name))
            elif '_ap_joint' in name:
                files_ap_np_comb.append(os.path.join(path, name))
            # if 'prep_myopic' in name:
            #     files_mp_prep.append(os.path.join(path, name))
            # elif 'prep_ap_self' in name:
            #     files_ap_prep_self.append(os.path.join(path, name))
            # elif 'prep_ap_other' in name:
            #     files_ap_prep_other.append(os.path.join(path, name))
            # elif 'prep_ap_joint' in name:
            #     files_ap_prep_comb.append(os.path.join(path, name))


    # No-Prep Myopic
    np_myopic = get_df_group_by_task(files_mp_np)
    print(np_myopic)
    np_myopic['label'] = 'No-Prep Myopic'

    print(f"No-Prep Myopic Task Cost (Avg): {np_myopic['avg_cost'].mean()}")

    # No-Prep Selfish Anticipatory Planning
    np_selfish = get_df_group_by_task(files_ap_np_self)
    np_selfish['label'] = 'No-Prep Selfish A.P.'

    print(f"No-Prep Selfish A.P. Task Cost (Avg): {np_selfish['avg_cost'].mean()}")

    # No-Prep Anticipatory Planning with joint expected cost
    np_proactive = get_df_group_by_task(files_ap_np_comb)
    np_proactive['label'] = 'No-Prep Proactive A.P'

    print(f"No-Prep Proactive A.P Task Cost (Avg): {np_proactive['avg_cost'].mean()}")

    # # Prep Myopic
    # prep_myopic = get_df_group_by_task(files_mp_prep)
    # prep_myopic['label'] = 'Prep Myopic'

    # print(f"Prep Myopic Task Cost (Avg): {prep_myopic['avg_cost'].mean()}")

    # # # Prep Selfish Anticipatory Planning
    # prep_selfish = get_df_group_by_task(files_ap_prep_self)
    # prep_selfish['label'] = 'Prep Selfish A.P.'

    # print(f"Prep Selfish A.P. Task Cost (Avg): {prep_selfish['avg_cost'].mean()}")

    # # # Prep Anticipatory Planning with joint expected cost
    # prep_proactive = get_df_group_by_task(files_ap_prep_comb)
    # prep_proactive['label'] = 'Prep Proactive A.P'

    # print(f"Prep Proactive A.P. Task Cost (Avg): {prep_proactive['avg_cost'].mean()}")

    # raise NotImplementedError
    #Combine and Plot
    # combined_df = pd.concat([np_myopic, np_selfish, np_proactive, prep_myopic, prep_selfish, prep_proactive])
    combined_df = pd.concat([np_myopic, np_proactive, np_selfish])
    print(combined_df)
    plot_tasks_comparison_cost(combined_df)
    raise NotImplementedError
    # df = pd.DataFrame(columns=["Method", "Value", "Error"])
    # # # No Preparation
    # np_mp_fail, se_0 = process_task_files(files_mp_np)
    # np_ap_self_fail, se_1 = process_task_files(files_ap_np_self)
    # # np_ap_other_fail, se_2 = process_task_files(files_ap_np_other)
    # np_ap_comb_fail, se_3 = process_task_files(files_ap_np_comb)
    # i = 0
    # df.loc[i] = ["NP Myopic", np_mp_fail, se_0]
    # i += 1
    # df.loc[i] = ["NP AP (SELF)", np_ap_self_fail, se_1]
    # i += 1
    # # df.loc[i] = ["NP AP (OTHER)", np_ap_other_fail, se_2]
    # # i += 1
    # df.loc[i] = ["NP AP (COMB)", np_ap_comb_fail, se_3]
    # i += 1
    # prep_mp_fail, se_0 = process_task_files(files_mp_prep)
    # prep_ap_self_fail, se_1 = process_task_files(files_ap_prep_self)
    # # prep_ap_other_fail, se_2 = process_task_files(files_ap_prep_other)
    # prep_ap_comb_fail, se_3 = process_task_files(files_ap_prep_comb)
    # df.loc[i] = ["Prep Myopic", prep_mp_fail, se_0]
    # i += 1
    # df.loc[i] = ["Prep AP (SELF)", prep_ap_self_fail, se_1]
    # i += 1
    # # df.loc[i] = ["Prep AP (OTHER)", prep_ap_other_fail, se_2]
    # # i += 1
    # df.loc[i] = ["Prep AP (COMB)", prep_ap_comb_fail, se_3]
    # print(df)
    # box_plot(df)
    # # raise NotImplementedError
    # prep_cost_myopic = process_cost(files_mp_prep)
    # prep_cost_self = process_cost(files_ap_prep_self)
    # # prep_cost_other = process_cost(files_ap_prep_other)
    # prep_cost_combine = process_cost(files_ap_prep_comb)
    # # print(prep_cost_myopic["avg_cost"].mean())
    # print(f"Prep & Myopic: {prep_cost_myopic['avg_cost'].mean()}")
    # print(f"Prep & Selfish: {prep_cost_self['avg_cost'].mean()}")
    # # print(f"Prep & Other: {prep_cost_other['avg_cost'].mean()}")
    # print(f"Prep & Combine: {prep_cost_combine['avg_cost'].mean()}")

def get_args():
    parser = argparse.ArgumentParser(
        description='Result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    compare(args)
