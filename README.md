# 🔍 Authorship Attribution - Digital Forensic in Victorian Era

## About the Project
This project explores authorship attribution for Victorian-era texts, focusing on automatically predicting which author wrote a given passage using stylometric analysis and machine learning. The goal is to show that an author’s “writing fingerprint” can be captured through quantitative features, even when texts share similar genre, time period, and language style.

Using a collection of prose from major Victorian authors (e.g., Charles Dickens, Arthur Conan Doyle, Thomas Hardy, the Brontë sisters, etc.), the project builds an end-to-end pipeline: from text preprocessing and feature extraction to model training, evaluation, and interactive inference via a simple app. The system learns from both classical stylometric features (sentence length, word length, punctuation patterns, function-word usage, etc.) and character/word n-grams, which have been shown to work well for author identification.

On top of this, several machine learning models are trained and compared—such as Logistic Regression, Linear SVM, and Random Forest—to identify which approach best distinguishes between Victorian authors. Model performance is evaluated using metrics like accuracy, macro F1-score, and confusion matrices to see which authors are most often confused with each other.

Beyond pure accuracy, the project highlights practical applications in:

- Digital forensics & cybercrime – adapting similar techniques to detect the author of anonymous online texts or suspicious documents.

- Literary studies & digital humanities – supporting scholars in authorship debates and stylistic analysis.

- Plagiarism and ghostwriting detection – identifying writing style inconsistencies across documents.

Overall, this project demonstrates how mathematics, NLP, and machine learning can be combined to analyze Victorian literature and recover hidden patterns of authorship—bridging classic literature with modern AI methods.

## About the Dataset

### Link Project: https://stylomedetective.streamlit.app/

### Source
Gungor, A. (2018). Victorian Era Authorship Attribution [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SW4H.
