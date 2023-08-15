

# Take in a retrieval store
# Take in dataset of question chunk pairs
# Take in model
# retrieve chunks given question
# check if true chunk is in retrieved chunks

from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

class RetrievalEvaluator:
    def __init__(self, retrieval_store, model):
        self.retrieval_store = retrieval_store
        self.model = model

    def evaluate(self, evaluation_dataset, k_retrievals=3):
        self.results = {i: 0 for i in range(1, k_retrievals+1)}
        for i in tqdm(range(len(evaluation_dataset)), total=len(evaluation_dataset), desc="Evaluating"):
            question = evaluation_dataset.get_text(i)["question"]
            chunk = evaluation_dataset.get_text(i)["chunk"]
            retrieved_chunks = self.retrieval_store.query(question, self.model, k_retrievals)["documents"][0]
            for k in range(k_retrievals, 0, -1):
                if chunk not in retrieved_chunks[:k]:
                    break
                self.results[k] += 1
        for k in self.results.keys():
            self.results[k] /= len(evaluation_dataset)
        return self.results

    def plot_performance(self, save_path=None):
        df = pd.DataFrame(list(self.results.items()), columns=['Top-k', 'Performance'])
        
        # Initialize the matplotlib figure
        f, ax = plt.subplots(figsize=(6, 6))
        
        # Plot the total retrievals
        sns.set_color_codes("pastel")
        sns.barplot(x='Top-k', y='Performance', data=df, color="b")
        
        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="lower right", frameon=True)
        ax.set(ylim=(0, 1), xlabel="Top-k retrievals", ylabel="Performance",title="Top-k Retrievals Performance")
        sns.despine(left=True, bottom=True)

        for index, row in df.iterrows():
            ax.text(row.name, row['Performance'], round(row['Performance'], 2), color='black', ha="center")
        
        if save_path is None:
            return f
        else:
            plt.savefig(save_path)