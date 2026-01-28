"""
File: missionimtextible.py (inspired by mission impossible)
Authors: Sara and Alison
Description: extension of textastic.py
"""
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


class TextAnalysisFramework:
    """
    A framework for analyzing and visualizing text.
    Loads txt files, analyzes word frequencies, and creates three separate visualizations.
    """

    def __init__(self):
        """ Initialize the framework with empty data structures."""
        self.data = defaultdict(dict)
        self.stop_words = set()

    # Needed to add a method that normalized words because we were seeing issues like
    # "children" and "children's" in separate categories when they should be combined
    @staticmethod
    def normalize_word(word):
        """
        Normalize words in the docs by removing plurals
        :param word: str, OG word
        :return: str, Normalized word
        """

        # Remove 's
        if word.endswith("'s"):
            word = word[:-2]

        # Remove s'
        elif word.endswith("s'"):
            word = word[:-2]

        return word

    def default_parser(self,filename):
        """
        Processing plain text .txt and removes unnecessary punctuation and also converts everything into lowercase
        :param filename: .txt file
        :return: word frequency counter = 'wordcount', total number of words = 'numwords', OG text = 'text'
        """
        with open(filename, 'r', encoding='utf-8') as f:
            text=f.read()

        # Removes punctuation
        text_no_punctuation = text.replace('.','').replace(',','')

        # Makes text lowercase and split into words
        text_lower = text_no_punctuation.lower()
        words = text_lower.split()

        normalized_words = [self.normalize_word(word) for word in words]

        if self.stop_words:
            normalized_words = [word for word in normalized_words if word not in self.stop_words]

        # Measures how often each word is mentioned
        word_frequency = Counter(normalized_words)

        results = {
            'wordcount' : word_frequency,
            'numwords' : len(normalized_words),
            'text' : text
        }
        return results

    def load_stop_words(self, stopfile):
        """
        Words we're filtering from the files and from our data (stop words)
        :param stopfile = file containing stop words
        :return: None
        """
        try:
            with open(stopfile, 'r', encoding='utf-8') as f:
                # utf-8 handles special characters
                self.stop_words = set(self.normalize_word(word.strip().lower()) for word in f.readlines())
            print(f"Loaded {len(self.stop_words)} stop words")
            # had to look more into encoding using https://www.jetbrains.com/help/pycharm/encoding.html

        except FileNotFoundError:
            print("stop word file not found")

    def load_text(self, filename, label=None, parser=None):
        """
        Load the text file into the framework
        :param filename: Text file
        :param label: The document label
        :param parser: The custom parser
        :return: None
        """

        # Uses default parser if none is provided
        if parser is None:
            results = self.default_parser(filename)
        else:
         results = parser(filename)

        # Uses filename for the label if none is provided
        if label is None:
            label = filename

        # Puts results in the data dict
        for k, v in results.items():
         self.data[k][label] = v

    def wordcount_sankey(self, word_list=None,k=5):
        """
        create a sankey diagram mapping texts to words
        :param word_list: list, and it's the specific words we're looking at
        :param k: int, number of top words per text if the list isn't given
        :return: None
        """
        wordcount = self.data['wordcount']

        if not wordcount:
            print("there is no text! unable to process")
            return

        # If no list is provided, get top k words instead
        if word_list is None:
            top_words = set()
            for label, word_frequency in wordcount.items():
                top_k = [word for word, count in word_frequency.most_common(k)]
                top_words.update(top_k)
            word_list = sorted(top_words)

        labels = list(wordcount.keys()) + word_list

        sources = []
        targets = []
        values = []

        for i, (text_labels, word_frequency) in enumerate (wordcount.items()):
            for w in word_list:
                if w in word_frequency:
                    sources.append(i)
                    targets.append(len(wordcount) + word_list.index(w))
                    values.append(word_frequency[w])

        # Sankey diagram
        fig = go.Figure(data=go.Sankey(
            node=dict(
                pad=10,
                thickness=20,
                line=dict(color="green", width = 0.5),
                label = labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        ))

        fig.update_layout(
            title = f"text to word flow (top {k} words per text)",
            font_size = 12
        )

        # Open sankey in a new browser
        import webbrowser
        fig.write_html("sankey_diagram.html")
        webbrowser.open("sankey_diagram.html")
        print("Sankey diagram saved as 'sankey_diagram.html' and opened in browser")


    def subplot_bars(self, k=8):
        """
        Create a visual of several bar charts displaying the top words from each document
        :param k: Most frequent words displayed in each mission statement (top 8)
        :return: None
        """
        documents = list(self.data['wordcount'].keys())
        num_docs = len(documents)

        if num_docs == 0:
            raise ValueError("No documents are loaded!")

        # Grid dimensions
        cols = min(3, num_docs)
        rows = (num_docs + cols - 1) // cols

        # Create subplot
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5*rows))

        # Added ravel so that the 1d array indexes properly
        # https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
        if num_docs == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        # Create bar chart for each document
        for idx, document in enumerate(documents):
            wc = self.data['wordcount'][document]
            top_k = wc.most_common(k) # used "most_common" from Counter that returns the top 8 k

            words = [w for w, _ in top_k]
            counts = [c for _, c in top_k]

            ax = axes[idx]

            ax.bar(words, counts)
            ax.set_title(document, fontsize=11, pad=8)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.tick_params(axis='x', labelsize=8, rotation=45)
            ax.tick_params(axis='y', labelsize=8)

            # We needed to make more space and make the graph less cluttered so we removed the spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)

        # Some subplot axes were blank so we needed to remove out of range ones
        for extra in range(len(documents), len(axes)):
            axes[extra].axis('off')

        fig.tight_layout()
        plt.show()

    def heatmap(self,top_n=15):
        """
        Create a heatmap display showing the most frequently used words across all documents
        :param top_n: int, The amount/number of top words that must be included
        :return: None
        """

        wordcount=self.data['wordcount']

        if not wordcount:
            print("NO TEXT LOADED")
            return

        # Aggregate all words
        all_words = Counter()

        for word_frequency in wordcount.values():
            all_words.update(word_frequency)

        # Top "n" most common words
        top_words = [word for word, count in all_words.most_common(top_n)]
        labels = list(wordcount.keys())

        # Documents and words matrix that measures frequency
        matrix = []
        for label in labels:
            row = [wordcount[label].get(word, 0) for word in top_words]
            matrix.append(row)

        matrix = np.array(matrix)

        # Create heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(top_words)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(top_words, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Word Frequency', rotation=270, labelpad=20, fontweight='bold')

        ax.set_title(f'Word Frequency Heatmap (Top {top_n} Words)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Words', fontweight='bold', fontsize=12)
        ax.set_ylabel('Companies', fontweight='bold', fontsize=12)

        plt.tight_layout()
        plt.show()
