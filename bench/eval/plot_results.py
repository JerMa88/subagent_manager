import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def plot_results(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                row = json.loads(line)
                
                # Hierarchical stats
                hier_speed = row['hier_tokens'] / row['hier_time'] if row['hier_time'] > 0 else 0
                hier_avg_tokens = row['hier_tokens'] / max(row.get('hier_subtasks', 1), 1)
                
                # Baseline stats
                base_speed = row['base_tokens'] / row['base_time'] if row['base_time'] > 0 else 0
                base_avg_tokens = row['base_tokens'] / max(row.get('base_subtasks', 1), 1)
                
                # Accuracy stats
                hier_acc = 1 if row['hier_patch'] else 0
                base_acc = 1 if row['base_patch'] else 0

                data.append({
                    'Method': 'Hierarchical',
                    'Accuracy': hier_acc,
                    'Time (s)': row['hier_time'],
                    'Speed (tokens/s)': hier_speed,
                    'Total Tokens': row['hier_tokens'],
                    'Avg Tokens per Agent': hier_avg_tokens
                })
                data.append({
                    'Method': 'Baseline',
                    'Accuracy': base_acc,
                    'Time (s)': row['base_time'],
                    'Speed (tokens/s)': base_speed,
                    'Total Tokens': row['base_tokens'],
                    'Avg Tokens per Agent': base_avg_tokens
                })
            except Exception as e:
                print(f"Error parsing line: {e}")
                
    if not data:
        print("No data found!")
        return
        
    df = pd.DataFrame(data)
    
    # 1. Accuracy Bar Plot
    plt.figure(figsize=(8, 5))
    acc_df = df.groupby('Method')['Accuracy'].mean().reset_index()
    sns.barplot(data=acc_df, x='Method', y='Accuracy', palette='Set2')
    plt.title('Accuracy (Patch Generated %)')
    plt.savefig('bench/results/accuracy_bar.png')
    plt.close()
    
    # Plotting lists
    metrics = [
        ('Time (s)', 'Total Time to Finish (s)'),
        ('Speed (tokens/s)', 'Token Generation Speed (tokens/s)'),
        ('Total Tokens', 'Total Tokens Generated'),
        ('Avg Tokens per Agent', 'Average Tokens per Agent')
    ]
    
    # Generate Box and Violin plots
    for metric, title in metrics:
        # Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Method', y=metric, palette='Set2')
        plt.title(f'Box Plot: {title}')
        plt.savefig(f'bench/results/{metric.split()[0]}_box.png')
        plt.close()
        
        # Violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='Method', y=metric, palette='Set2')
        plt.title(f'Violin Plot: {title}')
        plt.savefig(f'bench/results/{metric.split()[0]}_violin.png')
        plt.close()

    print("Plots saved in bench/results/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='bench/results/compare_300.jsonl')
    args = parser.parse_args()
    plot_results(args.input)
