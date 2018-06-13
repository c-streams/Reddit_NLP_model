# Using Natural Language Processing and Machine Learning to predict the level of interaction on Reddit posts 

## Background

Reddit is a popular social media platform and is the 6th most visited website in the world [1]. It averages 542 million monthly visitors who use the site as a news source, content rating platform, and discussion board [1]. Reddit's default homepage is the hot list, which is the site's most popular posts. The site is structured into subreddits for subject specific content which users can subscribe to. Users can engage with posts in various ways, including commenting, sharing, and up or down voting.  A sample post is shown in Figure 1 below.



Figure 1: Sample Reddit post with labeled features 

With over 234 million users, Reddit has the power to spread content to the masses [1]. This begs the question, what characteristics of a post make it popular or not?

## Dataset

In order to answer this question, I needed to refine it into a data science problem. Namely, what characteristics of a Reddit post are most predictive of the overall interaction on a thread (as measured by number of comments)? I will turn this into a classification problem by creating two classes of posts, those with a high number of comments (above the median) and those with a low number of posts (at or below the median). 

Using Reddit's open source API, I scraped over 15,000 posts on the hot list over a 7 day period [2, 3]. While there are many features on each post, I decided to select 4 to model: post title, subreddit, time posted, and number of comments. Since I wanted to look at age of the post in hours, I had to feature engineer a new column that represented the difference between the time created and the time I scraped the data and then convert to hours. I also had to create my target column, to assign each post as having a high or low number of comments based on the median. The resulting dataset had no missing values and required minimal cleaning.

## Exploratory Data Analysis 

Since the dataset contains very few numerical columns, there was little EDA I could perform without Natural Language Processing. I therefore decided to look at the distribution of the age of posts (Figure 2). Age was right skewed and revealed the shelf life for the amount of time a post can be on the hot list as 24 hours.



Figure 2: Distribution of the age of posts 

I also grouped the posts by subreddit and examined the mean number of comments per post (Figure 3). This revealed subreddits which are much more active than others. The 10 subreddits listed in Figure 3 are also some of the most subscribed to subreddits on Reddit [4].



Figure 3: Most active subreddits on hot list 

Even before modeling, it would suggest that posting in a popular subreddit will be important to getting a high number of comments.  I will keep this in mind when modeling. 

## Model Selection 

I decided to compare several Bag of Words NLP methods with various tree and non tree based classification machine learning models. CountVectorizer is the simplest of bag of words methods, where text data is translated into numerical data based on the frequency each word appears in a document (Figure 4). TF-IDF is similar except it penalizes common words and gives rare words more influence. I found TF-IDF to give comparable results to CountVectorizer when paired with the same models.



Figure 4: Bag of Words NLP explanation 

I also wanted to compare several tree and non tree based models using these different NLP methods with different features. For the tree based models, I used a simple DecisionTreeClassifer and a tuned RandomForest. For non-tree based models, I used tuned LogisticRegression, KNN, and Multinomial Naive Bayes. For each of these algorithms, I made models using title only, subreddit and age only, and title, subreddit and age as predictors.

Models using only title as the feature performed the worst. It would appear that title keywords are not very indicative of the number of comments a post will get. The two best models used title, subreddit, and age as predictors, TF-IDF pre-processing, and either a tuned logistic regression or tuned random forest algorithm. For the purposes on interpretability, I will focus on the logistic regression model here.

## Modeling and Evaluation 

The best performing logistic regression model used a Ridge penalty and generalized well to new data after GridsearchCV cross validated optimal penalty and C hyperparameters. This model, and others, were very overfit before tuning, as evidenced by high train accuracy scores and significantly lower test accuracy scores. The model had a 77% accuracy score on the test data (Figure 5).



Figure 5: Confusion matrix of best logistic regression model 

After examining the beta coefficients of the model, it became clear that subreddit is the strongest indicator for how popular a post will be. This makes sense since some subreddits have more followers and thus a higher activity than others. In addition the odds of having a large number of comments increase the longer the post is on Reddit. Again, these insights confirm common sense. The top 5 predictors, all subreddits, are gaming, FortNiteBR, pics, funny, and todayilearned (Figure 6). These subreddits, except FortNiteBR, are all in the top 20 most subscribed to subreddits [4]. In the case of FortNiteBR, which is a subreddit for the popular video game FortNite, I believe that my model might be influenced by external news in the particular week my data was scraped. I would need to collect posts over a longer period to correct for this.



Figure 6: Top 5 predictors of logistic regression model 

So what makes a reddit post popular, or at least highly commented? Post to popular subreddits with the most subscribers like funny, FortNiteBR, and gaming subreddits to increase your odds of creating a highly commented post.

## Next Steps

In terms of next steps, I would like to look at additional post features such as voting, shares, comment content, and post date and time in order to narrow down a recommendation. I would also like to change the target classification thresholds rather than just using the median since there is a wide range of comments on posts. Lastly, I would like to actually create a post based on my recommendations to test the outcome. 


## References

[1] https://www.alexa.com/siteinfo/reddit.com

[2] http://www.reddit.com/hot.json

[3] https://github.com/reddit-archive/reddit/wiki/JSON

[4] http://redditmetrics.com/top
