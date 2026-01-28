"""
File: missionimtextible_app.py
Authors: Sara and Alison
Description: extension of textastic_app.py
"""

import os
from missionimtextible import TextAnalysisFramework

DIR = "company_txt"

def main():
    """
    Main function is called in order to use the mission statements from company_txt directory
    - It loads the txt files and displays three visualizations: text-to-word sankey, subplots, and heatmap
    :return: None
    """
    # Init the framework
    ms = TextAnalysisFramework()

    # Loads stop words which takes out common words not relevant to the analysis
    ms.load_stop_words('stopwords.txt')

    files_loaded = 0
    stopword_file = 'stopwords.txt'

    # Load all the txt files
    for fname in os.listdir(DIR):
        # Load all the txt files except for the stopwords file
        if fname.endswith(".txt") and fname != stopword_file:
            fullpath = os.path.join(DIR, fname)
            label = fname.replace(".txt", "")
            ms.load_text(fullpath, label=label)

    # Create visualizations
    ms.wordcount_sankey(k=5)
    ms.subplot_bars(k=8)
    ms.heatmap(top_n=8)

if __name__ == "__main__":
    main()

