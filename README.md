# Sentiment-Analysis on Amazon Product Reviews

For this project, I decide to mine data from *Amazon* site and collect the reviews of a pair of headphones by using the Object-Oriented Programming (OOP).

***Product*** *URL:* https://www.amazon.com/TOZO-T6-Bluetooth-Headphones-Waterproof/dp/B07RGZ5NKS/ref=cm_cr_arp_d_product_top?ie=UTF8 

***Review*** *URL:* https://www.amazon.com/product-reviews/B07RGZ5NKS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=

It consists of five jupiter notebooks:

* **1. Web Scraping:** the *class* ***Reviews*** provides methods to grab and extract reviews from an Amazon product page, store the scraped data and save it to a JSON file for further analysis.
* **2. Exploratory Data Analysis:** consists of Text *Pre-processing* techniques and Text *Exploratory* analysis.
* **3. Polarity Analysis:** evaluates two of the most common approaches, such as **VADER** Sentiment Scoring and **ROBERTA** Pre-Trained Model.
* **4. Feature Analysis:** Word-Vectorization techniques (**BoW**, **TF-IDF**) and Feature *Selection*.
* **5. Modelling:** consists of *splitting* dataset into train and test set, and *selection* of the most common machine learning models. 
