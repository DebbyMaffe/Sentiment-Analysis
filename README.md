# Web-Scraping and Sentiment-Analysis
**Web-Scraping and Sentiment-Analysis on Amazon Product Reviews** - University Project

**DATA SOURCE**

For this project, I decide to mine data from *Amazon* site and collect the reviews of a pair of headphones by using the Object-Oriented Programming (OOP).

***Product*** *URL:* https://www.amazon.com/TOZO-T6-Bluetooth-Headphones-Waterproof/dp/B07RGZ5NKS/ref=cm_cr_arp_d_product_top?ie=UTF8 

***Review*** *URL:* https://www.amazon.com/product-reviews/B07RGZ5NKS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=

It consists of **five** jupiter notebooks. 

# 01. WEB SCRAPING
**Web scraping** is the automated process of *extracting* data from websites, involving fetching web pages, parsing HTML or XML content, and extracting desired information without manual intervention. In a specific project, data is mined from *Amazon*, focusing on collecting reviews for a pair of headphones using Object-Oriented Programming (**OOP**). The scraping process involves careful consideration of **HTTP** *headers*, specifically the **User-Agent** and language preference, to avoid being blocked during data extraction.

The **Class Reviews** provides methods to grab and *extract* reviews from an Amazon product page, *store* the scraped data and *save* it to a JSON file for further analysis. Key functionalities of the class include:
* Initialization using the **init()** method, which sets up the object with the Amazon Standard Identification Number (**ASIN**), creates an *HTTP session*, defines *headers*, and constructs the *review URL*.
* **pagination()** method for handling *page requests* and checking for the *availability* of reviews on a given page.
* **get_reviews()** method for *extracting* relevant information such as title, rating, place, date, and body text from HTML elements representing reviews.
* **save()** method for *storing* the extracted review data in a JSON file named 'ASIN_ID_reviews.json'.
  
The dataset used for further analysis is a '**reviews.csv**' file containing **100 rows**.

# 02. EXPLORATORY DATA ANALYSIS
The Exploratory Data Analysis (**EDA**) for sentiment analysis involves several *pre-processing* steps to prepare the text data for numerical analysis. Key steps include **label encoding** for sentiment categories (negative, neutral, positive), **text pre-processing** techniques such as expanding contractions, lowercasing, removing punctuation and stopwords, and converting numbers into words. 
The pre-processed data is saved in a new DataFrame called '**rws_clean**'. 

Therefore, *Tokenization*, *part-of-speech* tagging, *stemming*, and *lemmatization* are performed to further refine the text data. A new DataFrame named '**rws_clean_tag**' is created to store the results of these operations.

The analysis then extends to examining text *length* distributions for different sentiment categories and creating **word clouds** based on pre-processed text. Additional insights are gained by exploring **word** *frequencies*, identifying frequent **bigrams** and **trigrams**, and visualizing the **top 20** words, bigrams, and trigrams in Amazon reviews. The same text *analysis* procedures are applied to each **sentiment** category separately, providing a more detailed understanding of word usage *patterns* in negative, neutral, and positive reviews. The results are visually represented to highlight the most frequent words, bigrams, and trigrams for each sentiment category.

# 03. POLARITY ANALYSIS
**Polarity Analysis** is a specific type of sentiment analysis that focuses on determining whether the sentiment expressed in the text is *positive*, *negative*, or *neutral*, assigning a polarity score that usually ranges from -1 to 1. In particular: 
* **Positive** Polarity: if the polarity score is close to **1**, indicating a *positive* sentiment. 
* **Negative** Polarity: if the polarity score is close to **-1**, revealing a *negative* sentiment. 
* **Neutral** Polarity: if the polarity score is close to **0**, representing a *neutral* sentiment. 

Since it can be performed using various techniques, including *rule-based* methods, *machine learning-based* methods, and *pre-trained* models, let's analyse two of the most common approaches:
* **1. VADER Sentiment Scoring**, a rule-based sentiment analysis tool designed for social media text. It uses a **pre-built** *lexicon* of words with sentiment scores to calculate a **compound** *score*, representing overall sentiment polarity and intensity. The analysis includes visualizations of sentiment scores based on different Amazon *star ratings*.
* **2. ROBERTA Pre-Trained Model**, a state-of-the-art **pre-trained** language model based on the *transformer* architecture. It is trained on a massive amount of data to capture diverse *language* patterns. The analysis involves visualizing sentiment scores based on different *star ratings*.

**Comparisons** between VADER and ROBERTA are made, discussing their *advantages* and *limitations*. The accuracy, precision, recall, and F1-score metrics are computed for both models, providing insights into their **performance** on different *sentiment* classes. The results indicate that **VADER** performs better on the *negative* class, while **ROBERTA** shows *similar* trends across sentiment categories.

# 04. FEATURE ANALYSIS
For *preparing* text data for machine learning, **Feature Analysis** includes *splitting* the dataset into training and test sets, applying *label encoding* techniques, and exploring different *text vectorization* methods such as Bag-Of-Words (**BoW**) and Term Frequency-Inverse Document Frequency (**TF-IDF**).

**Bag-Of-Words (BoW)**

BoW involves **tokenizing** text into individual words and creating a *matrix* of word counts for each document.
The resulting matrix represents the **frequency** of each word in the text, treating each word *independently*.
The technique is simple but does **not** consider word *order* and may **lose** *semantic* information.

**TF-IDF**

Term Frequency-Inverse Document Frequency (**TF-IDF**) considers the **importance** of words within a document and across the entire corpus, assigning **higher** *weights* to words that are **rare** but *relevant*. It involves calculating:
* Term Frequency (**TF**) represents the *frequency* of word in a document out of the *total* number of words in that document;
* Document Frequency (**DF**) is the ratio between the number of documents containing a word (W) and the total number documents in the corpus. It represents the **proportion** of documents that contain a certain word (**W**).
* Inverse Document Frequency (**IDF**) is the **logarithm** applied on the *reciprocal* of DF, such that the **more** *common* a word is across all documents, the **lesser** its *relevance* is for the current text.
  
**Feature Selection**

Feature selection is the process of choosing **relevant** *features* to improve machine learning model *performance*: it involves **removing** *low-variance* features to reduce noise and improve efficiency.
Low-variance features might **not** carry *significant* information for classification tasks. Once discussed the implementation of these techniques, including encoding categorical columns, removing low-variance features, and exploring vocabulary, feature selection based on variance is applied with a *threshold* of **0.25**.

Despite considering removing low-variance features, the decision is made to **retain** *'Clean Title'* and *'Clean Content'* due to the small dataset size compromising representativeness.

# 05. MODELLING
**Model selection** involves choosing a suitable machine learning or statistical model from a set of candidates for accurate *predictions* on *unseen* data.
Since the choice of model significantly impacts *performance* and generalization ability, let's define the plausible algorithms for text classification, including Naive Bayes (**NB**), Multinomial Naive Bayes (**MNB**), Logistic Regression (**LR**), Support Vector Machine (**SVM**), k-Nearest-Neighbors (**kNN**), Decision Tree (**DT**), RandomForest (**RF**), Gradient Boosting (**GB**), and XGBoost (**XGB**).
To compare the performance of different models, let's consider the following metrics:
* **Accuracy** measures the proportion of all positive predictions out of all predictions. 
* **Precision** represents the proportion of correctly predicted positive instances out of all instances that the model predicted as positive.
* **Recall** indicates the proportion of correctly predicted positive instances out of all actual positive instances.
* **F1-Score** is the harmonic mean of precision and recall. 
* **Support** is the number of actual occurrences of each class in the dataset.

Therefore, evaluate each model's *performance* on the **entire** dataset to gain insights. Once performed **5-fold** Cross-Validation to assess model performance *robustly*, **split** the dataset into train and test set. Evaluating models on the **training** set for accuracy, micro-averaged precision, weighted average recall, and weighted average F1-score, Naive Bayes, Decision Tree, RandomForest, Gradient Boosting, and XGBoost models exhibit **high** *scores*. To assess generalization performance, let's evaluate models on the **testing** set, showing that Logistic Regression, Support Vector Machine, and k-Nearest-Neighbors models exhibit **high** *scores* on test set.

# 06. CONCLUSION
Most models that perform **well** on the *training* set also show **high** scores on the *test* set.
Exceptions include Decision Tree and Gradient Boosting on the test set, while Logistic Regression, Support Vector Machine, and k-Nearest-Neighbors models perform well on the test set.
