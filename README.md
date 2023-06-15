# Predicting-Recipe-Trends
<img src="https://img.sndimg.com/food/image/upload/w_621,h_349,c_fill,fl_progressive,q_80/v1/img/recipes/20/22/44/AU2lov1lQ8O9BU2Svopb_Thai%20Satay%20Noodles%20202244_final%202.jpg">
<figcaption style="opacity: 0.5;">source: food.com</figcaption>

by Rio Aguina-Kang (raguinakang@ucsd.edu) and Judel Ancayan (jancayan@ucsd.edu)

---

## Framing the Problem

In this project, our objective was to explore what variables would affect recipe complexity. In order to explore this idea, we built two regression models: a baseline linear regressor and a final decision tree regressor. 

The response variables for both of these models is complexity of the recipe, which we defined as the number of steps a recipe has. We believed that the number of steps was the best represenation for a recipe's complexity, because very simple recipe's (such as making a sandwhich) have a very low amount of steps (gathering ingredients and assembling the sandwhich). On the other hand, more complex recipe's(such as making a pizza) would have more steps(mixing ingredients, kneading dough, etc). The features used to predict the number of steps were the time the recipe took (in minutes), amount of calories, number of ingredients, and the year the recipe was released. Each of these features were hypothesized to have some kind of correlation with the number of steps in a recipe.

The predictions made by the models were then scored based on the regression coefficient, with higher coefficients relating towhich features have the greatest impact on recipe complexity. We chose the regression coefficient as a grading metric over the RMSE, because number of steps is an integer that is hard to perfectly guess. Therefore the RMSE will tend to be much higher, implying poor correlations, whereas the regression coefficient will be normalized to a number between 0 and 1. The data that we used to build these models underwent a cleaning process detailed in <a href='https://rioak.github.io/Recipe-Complexity-Trends/'>this project's</a> data cleaning section and shown below (note that only the columns relevent to this project are shown):

```py
print(unique_recipe.head().to_markdown(index=False))
```
|   minutes |   calories |   n_ingredients |   year |   n_steps |   average rating |
|----------:|-----------:|----------------:|-------:|----------:|-----------------:|
|        50 |      386.1 |               7 |   2008 |        11 |                3 |
|        55 |      377.1 |               8 |   2008 |         6 |                3 |
|        45 |      326.6 |               9 |   2008 |         7 |                3 |
|        45 |      577.7 |               9 |   2008 |        11 |                5 |
|        25 |      386.9 |               9 |   2008 |         8 |                5 |

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

In order to analyze the fairness of our final model, we performed a permutation test on our data. This test was run with 1000 trials at a significance of 0.05. The following hypotheses were used to lead this test:

- **Null Hypothesis**: The model's regression coefficient will be the same between high average ratin and low average rating, and any difference is due to random chance
- **Alternative Hypothesis**: The model's score is biased and its score depends on whether or not the recipe has a high or low average rating

In order to perform this permutation test, we split the data into two groups: recipes with a high rating (defined as having an average rating greater than 3.5) and recipes with a low rating(average rating of 3.5 or lower). We then used these groups to calculate regression coefficients on the final model (we used the regression coefficient as opposed to the RMSE for the same reason stated under framing the problem), and subtracted the regression coefficient. The resulting value is our observed statistic. 

The test statistics will be calculated in a similar manner:

- **Test Statistic**: Difference in Regression Coefficients based on Average Rating
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Regressino Coefficients of Recipes with High Average Rating</mtext>
  <mo>&#x2212;<!-- − --></mo>
  <mtext>Regressino Coefficients of Recipes with Low Average Rating</mtext>
</math> 

 The test statistics found by this permutation test are graphed below, with the red line representing the observed test statistic:


<iframe src="permutation.html" width=800 height=600 frameBorder=0></iframe>

The p-value is calculated to be 0.065, which fails to reject the null hypothesis at a significance of 0.05.


**Conclusion**

Since the permutation test failed to reject the null hypothesis, it is likely (although not definitive) that our model is fair.

---