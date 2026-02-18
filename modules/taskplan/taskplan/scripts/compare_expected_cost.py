import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import re

SUFFIX = '807'


def plot_tasks_comparison_cost(combined_df):
    # Find unique task numbers
    task_nums = combined_df['num'].unique()
    # Calculate the difference in avg_exp for each task between "No-Prep Myopic" and "Prep Myopic"
    differences = list()
    for task in task_nums:
        no_prep_avg_exp = combined_df[(combined_df['num'] == task) & (combined_df['label'] == "No-Prep Myopic")]['avg_cost'].values[0]
        prep_avg_exp = combined_df[(combined_df['num'] == task) & (combined_df['label'] == "No-Prep Anticipatory")]['avg_cost'].values[0]
        differences.append(((prep_avg_exp - no_prep_avg_exp
                             ) / no_prep_avg_exp)*100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the unique labels to plot each one
    labels = combined_df['label'].unique()

    for label in labels:
        # Subset the dataframe based on the label
        subset_df = combined_df[combined_df['label'] == label]

        # Sort by 'num' to ensure the lines are plotted correctly
        subset_df = subset_df.sort_values(by='num', key=lambda x: x.map(
            natural_sort_key))

        # Plot the mean line
        ax.plot(subset_df['num'], subset_df['avg_cost'], label=label + '(Expected Cost)')

        # Plot the shaded standard error
        ax.fill_between(subset_df['num'], subset_df['avg_cost'] - subset_df['std_err'], subset_df['avg_cost'] + subset_df['std_err'], alpha=0.1)
        if label == 'No-Prep Anticipatory':
            i = 0
            for x, y in zip(subset_df['num'], subset_df['avg_cost']):
                ax.text(x, y + 1, f'{round(differences[i])}%', fontsize=8)
                i += 1

    # Customize the plot
    ax.set_title('Change of Avg expected cost after completing each Task')
    ax.set_xlabel('Task')
    ax.set_ylabel('Average Expected Cost')
    ax.legend(title='Planners')
    # Show the plot
    save_file = '/data/figs/compare-expected-cost.png'
    plt.savefig(save_file, dpi=1200)


def natural_sort_key(s):
    """
    This function provides a key for sorting strings that contain numbers
    in a way that '2' comes before '10'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def process_task_files(files):

    dfs = []

    for file_path in files:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, sep="\s*\|\s*", engine='python', header=None)
        df = df.dropna(axis=1, how='all')
        df.columns = ['seq', 'desc', 'num', 'cost', 'ec_est']

        # Clean up the data by stripping whitespace and removing unnecessary characters
        df['seq'] = df['seq'].str.strip().str.split(':').str[1].str.strip()
        df['num'] = df['num'].str.strip().str.split(':').str[1].str.strip()
        df['cost'] = df['cost'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['ec_est'] = df['ec_est'].str.strip().str.split(':').str[1].str.strip().astype(float)

        dfs.append(df)

    # Calculate the average cost group by 'task num' and total cost group by 'task seq'

    merged_df = pd.concat(dfs, ignore_index=True)
    avg_cost_by_task = merged_df.groupby('num').agg({'ec_est': ['mean', 'sem']}).reset_index()
    avg_cost_by_task.columns = ['num', 'avg_cost', 'std_err']
    avg_cost_by_task_df = avg_cost_by_task.sort_values(
        by='num', key=lambda x: x.map(natural_sort_key))
    return avg_cost_by_task_df


def compare(args):
    root = args.save_dir
    # print(root)
    task_files_np_mp = list()
    task_files_np_ap = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'beta_50' in name:
                if 'no_prep_myopic' in path:
                    task_files_np_mp.append(os.path.join(path, name))
                if 'no_prep_ap' in path:
                    task_files_np_ap.append(os.path.join(path, name))

    np_myopic_tasks = process_task_files(task_files_np_mp)
    np_ant_tasks = process_task_files(task_files_np_ap)

    np_myopic_tasks['label'] = 'No-Prep Myopic'
    np_ant_tasks['label'] = 'No-Prep Anticipatory'
    print(np_myopic_tasks)
    # raise NotImplementedError
    combined_df = pd.concat([np_myopic_tasks, np_ant_tasks])
    plot_tasks_comparison_cost(combined_df)


def get_args():
    parser = argparse.ArgumentParser(
        description='Result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    compare(args)