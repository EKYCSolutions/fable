import argparse
#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#
parser = argparse.ArgumentParser(description='Visualize the class distribution of the annotation file.')
parser.add_argument('annoation_filepath', type=str, help='Path to annotation csv file.')
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.annoation_filepath)
    to_keep = [c for c in df.columns if c not in ['filename','person_id','xmin','ymin','xmax','ymax']]
    df = df[to_keep]
    #
    df = df.sum().reset_index()
    df = df.rename(columns={
        "index": "class",
        0: "count"
    })
    #
    plt.figure(figsize=(10, 5))
    sns.barplot(x='class', y='count', data=df, palette='pastel')

    plt.title("Class Distribution", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    plt.show()
