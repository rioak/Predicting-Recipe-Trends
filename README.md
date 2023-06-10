# Predicting-Recipe-Trends
<img src="https://img.sndimg.com/food/image/upload/w_621,h_349,c_fill,fl_progressive,q_80/v1/img/recipes/20/22/44/AU2lov1lQ8O9BU2Svopb_Thai%20Satay%20Noodles%20202244_final%202.jpg">
<figcaption style="opacity: 0.5;">source: food.com</figcaption>

by Rio Aguina-Kang (raguinakang@ucsd.edu) and Judel Ancayan (jancayan@ucsd.edu)

---

## Introduction

In this project, we explored the relationship between time and the complexity of recipes. Using a dataset from <a href="https://www.food.com/">food.com</a>, we gain access to 231,536 observations across 17 distinct features. To effectively evaluate recipe complexity, we utilized a key feature called "n_steps," which represents the number of steps required to prepare a recipe. Additionally, to investigate temporal trends, we considered the "submitted" date column, which encompasses recipe submissions spanning from 2008 to 2018. Finally, we utilized the "id" column to identify unique recipes within the dataset.

---

## Cleaning and Exploratory Data Analysis
**Data Cleaning**

To begin, we utilized two different csv files: a csv containing recipe data and a csv containing interaction (comments and ratings) data. We imported both of these csvs as dataframes, and merged the two using a left merge on recipe data. With the merged dataframe, we changed the rating column such that all of the "0" values became NaN values due to the nature of these values representing comments without ratings. Following this, we added a column to the dataframe called "average rating", which represented the average rating from 1 to 5 that the recipe recived. Additionally, we typecasted the "submitted" column (which is the date the recipe was submitted) and the "date" column (which is when the comment was submitted) from string type columns into pandas datetime columns. 

**Exploratory Data Analysis**

To be able to perform our exploratory data analysis, we grouped the dataframe by id, and used the "max" aggregate to preserve data per recipe, as well as ensure there were no duplicate recipes in the dataframe. The first few rows of the cleaned dataframe with all the listed changes is shown here (please note that the dataframe shown here is different from the dataframe used in the assessment of missingness. The dataframe used in that section is this dataframe before it was grouped by id, in order to preserve all of the separate interactions by recipe):

```py
print(unique_recipe.head()[["submitted","n_steps"]].to_markdown(index=True))
```

|     id | submitted           |   n_steps |
|:-------|---------------------|----------:|
| 275022 | 2008-01-01 00:00:00 |        11 |
| 275024 | 2008-01-01 00:00:00 |         6 |
| 275026 | 2008-01-01 00:00:00 |         7 |
| 275030 | 2008-01-01 00:00:00 |        11 |
| 275032 | 2008-01-01 00:00:00 |         8 |


*Univariate Analysis*

<iframe src="univariate_EDA.html" width=800 height=600 frameBorder=0></iframe>

The above graph charts the distribution of the number of steps each recipe has. The graph is right skewed, with nearly 90% of recipes requiring less than 20 steps.


*Bivariate Analysis*

<iframe src="bivariate_EDA.html" width=800 height=600 frameBorder=0></iframe>

This line chart highlights the observed relationship between the number of steps in a recipe and the year it was submitted. The plot displays a clear positive trend upwards from 2008 until 2016, where there is a drop from approximately 12.7 steps to 12 steps, but returns to an increasing relationship and rises past the original 2015 year average of ~12.7 steps to ~13.6 steps and beyond until 2018. These observations suggest a positive correlation between the two, showcasing that as the year increases, so does the average number of steps per recipe.

---

## Assessment of Missingness

**NMAR Analysis**

We believe that the "Ratings" column in the merged dataframe between recipe data and interaction data is Not Missing At Random (NMAR). This is because all of the missing values in that column were intentionally made missing if the original value was zero, as ratings can only be between numbers 1 and 5.


**Missingness Dependency**

Both missingness analyses were performed using the following dataframe, and the missingness of the "rating" column:

```py
print(average_food[["name","id","minutes","date","rating","user_id"]].head().to_markdown(index=False))
```

| name                                 |     id |   minutes | date                |   rating |          user_id |
|:-------------------------------------|--------|-----------|---------------------|----------|-----------------:|
| 1 brownies in the world    best ever | 333281 |        40 | 2008-11-19 00:00:00 |        4 | 386585           |
| 1 in canada chocolate chip cookies   | 453467 |        45 | 2012-01-26 00:00:00 |        5 | 424680           |
| 412 broccoli casserole               | 306168 |        40 | 2008-12-31 00:00:00 |        5 |  29782           |
| 412 broccoli casserole               | 306168 |        40 | 2009-04-13 00:00:00 |        5 |      1.19628e+06 |
| 412 broccoli casserole               | 306168 |        40 | 2013-08-02 00:00:00 |        5 | 768828           |

details about this dataframe are provided in the Data Cleaning section


**Analyzing the dependency of the missingness of the "rating" column and the "minutes" column**

In order to analyze the dependency of the minutes column, we performed a permutation test with a significance level of 0.01 and 100 trials. The following hypotheses were used to lead this test:

- **Null Hypothesis**: the missingness of the ratings column does not depend on the minutes of the recipe
- **Alternative Hypothesis**: the missingness of the ratings column does depend on the minutes of the recipe

The test statistic for this hypothesis was the absolute difference between the mean minutes of the ratings that are not missing subtracted by the mean minutes of the ratings that are missing. This is because if there is a significant difference between the two means, it would imply a relationship between the value of the "minutes" column and the missingness of the "rating" column.

- **Test Statistic**: Difference in means minutes of ratings
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>mean minutes of the ratings not missing</mtext>
  <mo>&#x2212;<!-- − --></mo>
  <mtext>mean minutes of the ratings missing</mtext>
</math> 

 The findings of the permutation test are summarized by the following graph, where the red line represents the observed test statistic:


<iframe src="minutes_missing.html" width=800 height=600 frameBorder=0></iframe>

The p-value for this permutation test ends up being 0.08, which results in failing to reject the null hypothesis at a significance of 0.01.

**Analyzing the dependency of the missingness of the "rating" column and the "date" column**

In order to analyze the dependency of the date column, we performed a permutation test with a significance level of 0.01 and 100 trials. The following hypotheses were used to lead this test:

- **Null Hypothesis**: the missingness of the ratings column does not depend on the date of the interaction
- **Alternative Hypothesis**: the missingness of the ratings column does depend on the date of the interaction

The test statistic for this hypothesis was the difference between the median date of the ratings that are not missing subtracted by the median date of the ratings that are missing. This is because if there is a significant difference between the two medians, it would imply a relation between the value of the "date" column and the missingness of the "rating" column.

- **Test Statistic**: Difference in median dates of ratings
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>median dates of the ratings not missing</mtext>
  <mo>&#x2212;<!-- − --></mo>
  <mtext>median dates of the ratings missing</mtext>
</math> 

The findings of the permutation test are summarized by the following graph, where the red line represents the observed test statistic:

<iframe src="date_missing.html" width=800 height=600 frameBorder=0></iframe>

The p-value for this permutation test ends up being 0.00, which results in rejecting the null hypothesis at a significance of 0.01.

**Conclusion**

While the missingness of the rating column does not seem to depend on the minutes column, it does seem to depend on the date column. This suggests the missingness of the rating column is potentially Missing At Random (MAR).



---

## Hypothesis Testing

<img src="https://i.gifer.com/1PQg.gif">

To answer the question of whether or not online recipes have shown have become my complex over the last 10 years, we define our null and alternative hypotheses as such:

- **Null Hypothesis**: The average number of steps in 2018 is *the same* as in 2008, and any difference is a result of random chance

- **Alternative Hypothesis**: The average number of steps in 2018 is *greater* than it was in 2008. The observed difference in our samples cannot be explained by random chance alone.

To estimate an increase in recipe complexity between 2008 and 2018, we use a test statistic of the difference in mean number of steps between 2018 and 2008.

- **Test Statistic**: Difference in group means
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>mean number of steps in 2018</mtext>
  <mo>&#x2212;<!-- − --></mo>
  <mtext>mean number of steps in 2008</mtext>
</math> 

Since the recipe steps in 2008 is a different group from that of 2018, we run a permutation test. In this test, we generate new data by shuffling the "n_steps" column within the data to compute a single test statistic, calculating the differences in random distributions between recipe steps in 2018 and 2008. We do this over the course of 1000 trials, creating a new test statistic and appending it to an array of previous test statistics at each iteration. We also compute an observed test statistic from the original data

Here, we plotted the distribution of test statistics, as well as the obvserved test statistic noted as a red line in the plot below:

<iframe src="hypothesis_test.html" width=800 height=600 frameBorder=0></iframe>

Using the array of test statistics, we calculated the p-value by averaging the number of test statistics greater than our observed value, resulting in a p-value of 0.

**Conclusion**

Since the p-value is 0 and lower than the significance level of 0.05, we reject the null hypothesis that the number of steps in 2008 and the number of steps in 2018 come from the same distribution.

---