# Predicting-Recipe-Trends
<img src="https://img.sndimg.com/food/image/upload/w_621,h_349,c_fill,fl_progressive,q_80/v1/img/recipes/20/22/44/AU2lov1lQ8O9BU2Svopb_Thai%20Satay%20Noodles%20202244_final%202.jpg">
<figcaption style="opacity: 0.5;">source: food.com</figcaption>

by Rio Aguina-Kang (raguinakang@ucsd.edu) and Judel Ancayan (jancayan@ucsd.edu)

---

## Framing the Problem

In this project, we explored the relationship between time and the complexity of recipes. Using a dataset from <a href="https://www.food.com/">food.com</a>, we gain access to 231,536 observations across 17 distinct features. To effectively evaluate recipe complexity, we utilized a key feature called "n_steps," which represents the number of steps required to prepare a recipe. Additionally, to investigate temporal trends, we considered the "submitted" date column, which encompasses recipe submissions spanning from 2008 to 2018. Finally, we utilized the "id" column to identify unique recipes within the dataset.

---

## Baseline Model

To create a baseline model that we could compare other models against, we developed a linear regression model to predict the number of steps (n_steps) for recipes using two features: the year the recipe was released and the number of calories. Our model incorporates various components from the scikit-learn library to create a pipeline that performs preprocessing steps and fits a linear regression model.

In terms of feature representation, we have identified two numerical key features in our model. The 'year released' feature is considered quantitative, while the 'calories' feature is also quantitative in nature. While time in of itself is usually considered a continuous variable, since we are building our model on strictly the year that these recipes were released instead of including days, minutes, or seconds, we are considering them as a discrete numerical variable. On the other hand, calories is a continuous variable, taking on forms within a certain range or interval, and they can have an infinite number of possible values between any two specific values. In addition, In the case of "calories," since calories can take on completely different forms depending on the kinds of foods a recipe is making, we decided to standardize calories for our model to ensure consistency.

Moving on to the performance of our model, the provided results indicate a train score of 0.02566655266310225 and a test score of 0.0282273280524431. As a group, we consider these scores to be very low and close in value, ultimately a "bad" model. This suggests that our current model is not performing well in capturing the variability in the target variable (n_steps) based on the given features. We understand that further evaluation and improvement efforts are required to enhance the predictive capabilities of our model.

---

## Final Model

The final model we developed aimed to improve the prediction of the number of steps (n_steps) for recipes. We added additional features to enhance the model's performance based on our understanding of the data generating process. The features we incorporated are 'minutes', 'calories', 'year', and 'n_ingredients'. Here's why we believe these features are beneficial for the prediction task:

- 'Minutes': We utilized the 'minutes' feature by applying a Binarizer transformation, categorizing it as "hour or more" or "less than an hour." This feature captures the duration of recipe preparation, which can potentially impact the number of steps required. Recipes with longer durations might involve more complex steps, leading to a higher number of steps.

- 'Calories': We included the 'calories' feature and standardized it using StandardScaler. Caloric content can provide insights into the complexity of recipes and the ingredients used. Recipes with higher caloric content might involve more elaborate steps or additional ingredients, leading to a higher number of steps.

- 'Year': We considered the 'year' feature, representing the year the recipe was released. The year of recipe publication can be an indicator of culinary trends, advancements in cooking techniques, and evolving recipe formats. By incorporating this feature, we capture potential variations in the number of steps over time.

- 'N_ingredients': We included the 'n_ingredients' feature, representing the number of ingredients in a recipe. More ingredients generally require additional preparation steps, such as chopping, peeling, or combining, which can influence the number of steps involved.

For the modeling algorithm, we chose the MLPRegressor (Multi-Layer Perceptron Regressor) from scikit-learn's neural_network module. This algorithm is capable of learning complex relationships between the features and the target variable. Through its architecture of multiple layers and nonlinear activation functions, it can capture intricate patterns in the data.

To determine the best hyperparameters, we employed a GridSearchCV approach. We specified a range of hyperparameters to explore, such as 'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate', 'learning_rate_init', 'max_iter', and 'early_stopping'. The GridSearchCV method exhaustively searches through the hyperparameter combinations and selects the best-performing set of hyperparameters based on the chosen scoring metric.

The final model's performance is an improvement over the baseline model, as it incorporates additional features and a more sophisticated modeling algorithm. The baseline model's performance was relatively low, with scores of 0.02566655266310225 for the train set and 0.0282273280524431 for the test set. The final model, with the optimized hyperparameters identified through GridSearchCV, aims to enhance the predictive capabilities by leveraging the added features and adjusting the neural network's architecture. The selected hyperparameters represent the best combination that maximizes the model's performance based on the chosen scoring metric.

---

## Fairness Analysis

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

...

- **Null Hypothesis**: ...
- **Alternative Hypothesis**: ...

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