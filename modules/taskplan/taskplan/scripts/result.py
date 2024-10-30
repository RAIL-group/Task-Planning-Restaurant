import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import re
import random

SUFFIX = '807'


def plot_total_cost(df1, df2, label_text='Prep', image_path=''):
    true_costs = df1
    exp_cost = df2
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)

    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('Sequence Costs Myopic (Without Preparation)', fontsize=6)
    plt.ylabel('Sequence Cost ' + label_text, fontsize=6)
    plt.title('Comparison of Myopic (Without Preparation) vs ' + label_text,
              fontsize=8)
    save_file = '/data/figs/' + str(image_path) + '.png'
    plt.savefig(save_file, dpi=1000)


# Find unique task numbers
# task_nums = combined_df['num'].unique()
# Calculate the difference in avg_exp for each task between "No-Prep Myopic" and "Prep Myopic"
# differences = list()
# for task in task_nums:
#     no_prep_avg_exp = combined_df[(combined_df['num'] == task) & (combined_df['label'] == "No-Prep Myopic")]['avg_cost'].values[0]
#     prep_avg_exp = combined_df[(combined_df['num'] == task) & (combined_df['label'] == "No-Prep Anticipatory")]['avg_cost'].values[0]
#     differences.append(((prep_avg_exp - no_prep_avg_exp
#                          ) / no_prep_avg_exp)*100)


def plot_tasks_comparison_cost(combined_df):
    color_map = {
        'No-Prep Myopic': 'cyan',
        'Prep Myopic': 'orangered',
        'No-Prep Anticipatory': 'gold',
        'Prep Anticipatory': 'fuchsia'
    }
    marker_map = {
        'No-Prep Myopic': 'o',
        'Prep Myopic': 'd',
        'No-Prep Anticipatory': 's',
        'Prep Anticipatory': '*'
    }
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
        ax.plot(subset_df['num'], subset_df['avg_cost'], label=label,
        marker=marker_map.get(label, '.'), color=color_map.get(label, 'black'))

        # Plot the shaded standard error
        add_noise = 0
        ax.fill_between(subset_df['num'],
                        subset_df['avg_cost'] - subset_df['std_err'] - add_noise,
                        subset_df['avg_cost'] + subset_df['std_err'] + add_noise,
                        color=color_map.get(label, 'black'), 
                        alpha=0.1)
        # if label == 'Prep Myopic':
        #     i = 0
        #     for x, y in zip(subset_df['num'], subset_df['avg_cost']):
        #         ax.text(x, y + 1, f'{round(differences[i])}%', fontsize=8)
        #         i += 1

    # Customize the plot
    # ax.set_title('Average Cost Per Task', fontsize=10)
    # ax.set_xlabel('Task Number', fontsize=10)
    # ax.set_ylabel('Average Expected Cost', fontsize=10)
    # ax.legend(title='Planners', fontsize=10)
    # ax.set_ylabel('Average Cost of Task Number')
    ax.legend(title='Planners')
    ax.margins(x=0)
    ax.set_xticklabels([])
    plt.tight_layout()
    # Show the plot
    save_file = '/data/figs/compare-average-cost-final.png'
    plt.savefig(save_file, dpi=1200, bbox_inches='tight')


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
    avg_cost_by_task = merged_df.groupby('num').agg({'cost': ['mean', 'sem']}).reset_index()
    avg_cost_by_task.columns = ['num', 'avg_cost', 'std_err']
    avg_cost_by_task_df = avg_cost_by_task.sort_values(
        by='num', key=lambda x: x.map(natural_sort_key))
    return avg_cost_by_task_df


def normalize_desc(desc):
    # Normalize the format of the description to remove digits
    x = re.sub(r"\('(\w+?)\d*', '(\w+?)\d*'\)", r"('\1', '\2')", desc)
    if 'clean' in x:
        x = 'cleaning_tasks'
    if 'place' in x or 'clear' in x:
        x = 'organizing_tasks'
    return x


def process_task_files_desc(files):
    dfs = []
    for file_path in files:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, sep="\s*\|\s*", engine='python', header=None)
        df = df.dropna(axis=1, how='all')
        df.columns = ['seq', 'desc', 'num', 'cost', 'ec_est']

        # Clean up the data by stripping whitespace and removing unnecessary characters
        df['seq'] = df['seq'].str.strip().str.split(':').str[1].str.strip()
        df['desc'] = df['desc'].str.strip().str.split(':').str[1].str.strip()
        df['num'] = df['num'].str.strip().str.split(':').str[1].str.strip()
        df['cost'] = df['cost'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['ec_est'] = df['ec_est'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['desc'] = df['desc'].apply(lambda x: normalize_desc(x))
        dfs.append(df)
    # Calculate the average cost group by 'task num' and total cost group by 'task seq'

    merged_df = pd.concat(dfs, ignore_index=True)
    avg_cost_by_task = merged_df.groupby('desc')['cost'].agg(
        ['mean', 'sem']).reset_index()
    avg_cost_by_task.columns = ['num', 'avg_cost', 'std_err']
    avg_cost_by_task_df = avg_cost_by_task.sort_values(
        by='num', key=lambda x: x.map(natural_sort_key))

    return avg_cost_by_task_df


def plot_bar_analysis(data):
    means = data.mean()
    standard_errors = data.sem()

    comparative_analysis = pd.DataFrame({
        'Mean': means,
        'Standard Error': standard_errors
    })

    # Exclude 'eval' as it's not an approach
    comparative_analysis = comparative_analysis.drop(index='eval')

    # Calculate the performance gain relative to the baseline
    baseline_mean = comparative_analysis.loc['np_myopic', 'Mean']

    # Calculate the percentage improvement for each approach relative to baseline
    comparative_analysis['Improvement (%)'] = ((baseline_mean - comparative_analysis['Mean']) / baseline_mean) * 100

    # Plotting the mean and standard error for each approach
    plt.figure(figsize=(12, 7))

    # Bar plot for mean values
    bars = plt.bar(comparative_analysis.index, comparative_analysis['Mean'], yerr=comparative_analysis['Standard Error'], capsize=5, color=['skyblue', 'salmon', 'limegreen', 'orange', 'purple'])

    # Adding titles and labels
    # plt.suptitle('Comparative Analysis of Approaches with Performance Gain Relative to NP-Myopic', y=1.05, fontsize=18)
    plt.title('Comparative Analysis of Approaches with Performance Gain Relative to NP-Myopic', fontsize=10)
    plt.xlabel('Approaches')
    plt.ylabel('Cost')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate the bars with the improvement percentage
    for bar, improvement in zip(bars, comparative_analysis['Improvement (%)']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 20, f'{improvement:.2f}%', ha='center', va='bottom')

    # Show the plot
    plt.tight_layout()
    save_file = '/data/figs/compared-results-oracle.png'
    plt.savefig(save_file, dpi=1200)


def process_text_file(files):
    # Read the file into a pandas DataFrame

    dfs = []

    for file_path in files:
        df = pd.read_csv(file_path, sep="\s*\|\s*", engine='python', header=None)
        df = df.dropna(axis=1, how='all')
        # df.columns = ['eval', 'oracle', 'baseline', 'np_myopic',
        #               'prep_myopic', 'np_ap', 'prep_ap']
        df.columns = ['eval', 'np_myopic', 'prep_myopic', 'np_ap', 'prep_ap']

        # Clean up the data by stripping whitespace and removing unnecessary characters

        df['eval'] = df['eval'].str.strip().str.split(':').str[1].str.strip().astype(int)
        # df['oracle'] = df['oracle'].str.strip().str.split(':').str[1].str.strip().astype(float)
        # df['baseline'] = df['baseline'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['np_myopic'] = df['np_myopic'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['prep_myopic'] = df['prep_myopic'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['np_ap'] = df['np_ap'].str.strip().str.split(':').str[1].str.strip().astype(float)
        df['prep_ap'] = df['prep_ap'].str.strip().str.split(':').str[1].str.strip().astype(float)

        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    # Calculate the average cost group by 'task num' and total cost group by 'task seq'
    nmyp_cost = merged_df['np_myopic'].mean()/40
    pmyp_cost = merged_df['prep_myopic'].mean()/40
    nap_cost = merged_df['np_ap'].mean()/40
    pap_cost = merged_df['prep_ap'].mean()/40
    # base_cost = merged_df['baseline'].mean()

    print(nmyp_cost, nap_cost, pmyp_cost, pap_cost)
    # merged_df.to_csv(args.save_dir + 'analyzed.csv', index=False)
    # plot_bar_analysis(merged_df)
    # plot_total_cost(merged_df['np_myopic'], merged_df['prep_myopic'], label_text='Myopic (with Preparation)', image_path='pm-vs-npm-comp')
    plot_total_cost(merged_df['np_myopic'], merged_df['np_ap'], label_text='Anticipatory Planning (without Preparation)', image_path='nap-vs-npm-comp' + str(SUFFIX))
    # plot_total_cost(merged_df['np_myopic'], merged_df['prep_ap'], label_text='Anticipatory Planning (with Preparation)', image_path='pap-vs-npm-comp' + str(SUFFIX))

    return merged_df


def plot_individual_tasks(df_prep, df_np):
    merged_df = pd.merge(df_prep, df_np, on='num', suffixes=('_ap', '_mp'))
    # Plotting the bar plot side by side
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(merged_df))

    print(merged_df)
    # raise NotImplementedError

    bar1 = plt.bar(index, merged_df['avg_cost_ap'], bar_width, label='Planning from Prepared State')
    bar2 = plt.bar([i + bar_width for i in index], merged_df['avg_cost_mp'], bar_width, label='Planning from Non-Prepared State')

    plt.xlabel('Task Type')
    plt.ylabel('Average Cost')
    plt.title('Which task type has more impact?')
    plt.xticks([i + bar_width / 2 for i in index], merged_df['num'], rotation=90)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    save_file = '/data/figs/compared-tasks-comp.png'
    plt.savefig(save_file, dpi=1200)


def compare(args):
    root = args.save_dir
    # print(root)
    task_files_np_mp = list()
    task_files_p_mp = list()
    task_files_np_ap = list()
    task_files_p_ap = list()
    task_files_base = list()
    seq_cost_files = list()
    seq_ext_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'eval_cost' in name:
                seq_cost_files.append(os.path.join(path, name))
            elif 'eval_ext' in name:
                seq_ext_files.append(os.path.join(path, name))
            elif 'beta' in name:
                if 'no_prep_myopic' in path:
                    task_files_np_mp.append(os.path.join(path, name))
                elif 'prep_myopic' in path: 
                    task_files_p_mp.append(os.path.join(path, name))
                elif 'no_prep_ap' in path: 
                    task_files_np_ap.append(os.path.join(path, name))
                elif 'prep_ap' in path:
                    task_files_p_ap.append(os.path.join(path, name))
                elif 'baseline' in path:
                    task_files_base.append(os.path.join(path, name))

    np_myopic_tasks = process_task_files(task_files_np_mp)
    prep_myopic_tasks = process_task_files(task_files_p_mp)
    np_ant_tasks = process_task_files(task_files_np_ap)
    prep_ant_tasks = process_task_files(task_files_p_ap)
    # prep_base_tasks = process_task_files(task_files_base)
    # print(np_myopic_tasks)
    # print(np_ant_tasks)
    seq_df = process_text_file(seq_cost_files)
    print(seq_df)

    np_myopic_tasks['label'] = 'No-Prep Myopic'  # str(seq_df['np_myopic'].mean())
    np_ant_tasks['label'] = 'No-Prep Anticipatory'  # str(seq_df['np_ap'].mean())
    prep_myopic_tasks['label'] = 'Prep Anticipatory'
    prep_ant_tasks['label'] = 'Prep Myopic'
    # prep_ant_tasks['label'] = 'Prep Anticipatory: ' + str(seq_df['prep_ap'].mean())
    # prep_base_tasks['label'] = 'Baseline (Next task is revealed): ' + str(seq_df['baseline'].mean())
    combined_df = pd.concat([np_myopic_tasks, np_ant_tasks,
                             prep_myopic_tasks, prep_ant_tasks])
    # print(combined_df)
    plot_tasks_comparison_cost(combined_df)
    np_myopic_tasks = process_task_files_desc(task_files_np_mp)
    prep_myopic_tasks = process_task_files_desc(task_files_p_mp)
    # np_ant_tasks = process_task_files_desc(task_files_np_ap)
    # prep_ant_tasks = process_task_files_desc(task_files_p_ap)
    # prep_base_tasks = process_task_files_desc(task_files_base)

    # print(np_myopic_tasks)
    # print(prep_myopic_tasks)
    # np_myopic_tasks.to_csv(args.save_dir + 'analyzed_np.csv', index=False)
    # prep_myopic_tasks.to_csv(args.save_dir + 'analyzed_prep.csv', index=False)
    plot_individual_tasks(prep_myopic_tasks, np_myopic_tasks)


def get_args():
    parser = argparse.ArgumentParser(
        description='Result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    compare(args)