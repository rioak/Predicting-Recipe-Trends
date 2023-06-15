# Predicting-Recipe-Trends
<img src="https://img.sndimg.com/food/image/upload/f_auto,c_thumb,q_55,w_1280,ar_16:9/v1/img/recipes/12/19/62/rImnjthbQVyfxmTOddl4_Tomato%20phyllo%20pizza%20121962-5.jpg">
<figcaption style="opacity: 0.5;">source: food.com</figcaption>

by Rio Aguina-Kang (raguinakang@ucsd.edu) and Judel Ancayan (jancayan@ucsd.edu)

---

## Framing the Problem

In this project, our objective was to explore the factors influencing recipe complexity. We constructed two regression models: a baseline linear regressor and a final multilayer perceptron (MLP) regressor.

The response variable in both models is the recipe complexity, which we defined as the number of steps required to complete the recipe. We selected the number of steps as a measure of complexity because simpler recipes tend to have fewer steps (such as making a sandwich), while more complex recipes involve multiple intricate steps (like making a pizza). Our hypothesis was that certain features would correlate with the number of steps in a recipe, and we aimed to investigate these relationships.

The features used to predict the number of steps were as follows:

- Minutes: The duration of the recipe in minutes.
- Calories: The estimated number of calories in the recipe.
- Ingredients: The count of ingredients used in the recipe.
- Year: The year when the recipe was released.

We believed that these features could potentially provide insights into the recipe complexity. The models were trained to predict the number of steps based on these input features.

To evaluate the models' performance, we utilized the regression coefficient as a scoring metric. The regression coefficient indicates the strength and direction of the relationship between each feature and the recipe complexity. We preferred this metric over the root mean square error (RMSE) since accurately predicting the exact number of steps can be challenging due to its discrete nature. By using the regression coefficient, we normalized the scores to a range between 0 and 1, facilitating comparison and interpretation of feature importance.

Prior to training the models, the data underwent a thorough cleaning process, as outlined in the data cleaning section of this project (refer to <a href='https://rioak.github.io/Recipe-Complexity-Trends/'>this project's</a> data cleaning section). The relevant columns utilized in this project were retained for analysis.

Both the baseline linear regressor and the final MLP regressor were fitted to the cleaned dataset. The models' predictions were evaluated based on their regression coefficients, allowing us to identify the features that exerted the most significant influence on recipe complexity.

Please note that the complete details of the data cleaning process and the regression coefficient analysis can be found in the referenced project's documentation.

The dataframe below represents the first 5 rows of the dataset we used:

```py
print(unique_recipe.head().to_markdown(index=False))
```

|   minutes |   calories |   n_ingredients |   year |   n_steps |   average rating |
|:----------|------------|-----------------|--------|-----------|-----------------:|
|        50 |      386.1 |               7 |   2008 |        11 |                3 |
|        55 |      377.1 |               8 |   2008 |         6 |                3 |
|        45 |      326.6 |               9 |   2008 |         7 |                3 |
|        45 |      577.7 |               9 |   2008 |        11 |                5 |
|        25 |      386.9 |               9 |   2008 |         8 |                5 |

---

## Baseline Model

To create a baseline model that we could compare other models against, we developed a linear regression model to predict the number of steps (n_steps) for recipes using two features: the year the recipe was released and the number of calories. Our model incorporates various components from the scikit-learn library to create a pipeline that performs preprocessing steps and fits a linear regression model.

In terms of feature representation, we have identified two numerical key features in our model. The 'year released' feature is considered quantitative, while the 'calories' feature is also quantitative in nature. While time in of itself is usually considered a continuous variable, since we are building our model on strictly the year that these recipes were released instead of including days, minutes, or seconds, we are considering them as a discrete numerical variable. On the other hand, calories is a continuous variable, taking on forms within a certain range or interval, and they can have an infinite number of possible values between any two specific values. Additonally, since calories can take on completely different values depending on the kinds of foods a recipe is making, we decided to standardize calories for our model to ensure consistency.

Moving on to the performance of our model, the provided results indicate the scores: 

- Train score: 0.027334136400747222
- Test score: 0.020162972134046497

(Note: testing set represents 20% of the data)

Ultimately, we consider these scores to be very low. Both scores are close in value regardless of testing or training sets, and we do not think this baseline model is a satisfactory ("good") fit for predicting the number of steps for a recipe. This suggests that our current model is not performing well in capturing the variability in the target variable (n_steps) based on the given features.

---

## Final Model

The final model we developed aimed to improve our prediction of the number of steps (n_steps) for recipes. 

**Feature Selection**

In addition to the original features used in the baseline model (minutes, calories, year, and n_ingredients), we introduced feature engineering steps to preprocess the data. The features we incorporated are 'minutes', 'calories', 'year', and 'n_ingredients'. Here's why we believe these features are beneficial for the prediction task:

- 'Minutes': We utilized the 'minutes' feature by applying a Binarizer transformation, categorizing it as "hour or more" or "less than an hour." This feature captures the duration of recipe preparation, which can potentially impact the number of steps required. Recipes with longer durations might involve more complex steps, leading to a higher number of steps.

- 'Calories': We included the 'calories' feature and standardized it using StandardScaler. Caloric content can provide insights into the complexity of recipes and the ingredients used. Recipes with higher caloric content might involve more elaborate steps or additional ingredients, leading to a higher number of steps.

- 'Year': We considered the 'year' feature, representing the year the recipe was released. The year of recipe publication can be an indicator of culinary trends, advancements in cooking techniques, and evolving recipe formats. By incorporating this feature, we capture potential variations in the number of steps over time. Additionally, our previous exploratory analyses in our <a href="https://rioak.github.io/Recipe-Complexity-Trends/">Recipe Complexity Trends</a> project suggested an increase in recipe steps over time.

- 'N_ingredients': We included the 'n_ingredients' feature, representing the number of ingredients in a recipe. More ingredients generally require additional preparation steps, such as chopping, peeling, or combining, which can influence the number of steps involved.

**Modeling Algorithm and Hyperparameter Tuning**

For our final model, we chose the MLPRegressor algorithm, a neural network-based regressor capable of capturing complex relationships in the data. The initial pipeline configuration included a hidden layer size of 20 neurons. We then conducted hyperparameter tuning using grid search to find the best combination of hyperparameters that maximizes the model's performance.

The hyperparameters explored during grid search included 'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate', and 'early_stopping'.

```py
hyperparams = {
    'regress__hidden_layer_sizes': [(10,), (20,)],
    'regress__activation': ['relu', 'tanh'],
    'regress__solver': ['lbfgs', 'adam'],
    'regress__alpha': [0.0001, 0.001],
    'regress__learning_rate': ['constant', 'adaptive'],
    'regress__early_stopping': [True, False]
}
```

By searching over different combinations of these hyperparameters, we aimed to identify the optimal configuration that produces the best predictive performance.

**Grid Search Results and Best Hyperparameters**

The grid search was conducted, and the best hyperparameters were determined based on the mean squared error scores. The best hyperparameters found were as follows:

- Activation function: ReLU
- Alpha (L2 penalty parameter): 0.0001
- Early stopping: True
- Hidden layer sizes: (10,)
- Learning rate: Constant
- Solver: LBFGS

These hyperparameters represent the optimal configuration identified through grid search, aiming to minimize the model's error and enhance its predictive capabilities.

The final model's performance is an improvement over the baseline model, as it incorporates additional features and a more sophisticated modeling algorithm. The baseline model's performance was relatively low, with scores of 0.027 for the train set and 0.020 for the test set. The final model, with the optimized hyperparameters identified through GridSearchCV, aims to enhance the predictive capabilities by leveraging the added features and adjusting the neural network's architecture. The selected hyperparameters represent the best combination that maximizes the model's performance based on the chosen scoring metric.

**Final Model Performance**

Using the best hyperparameters obtained from the grid search, we rebuilt the pipeline with the best hyperparameters. The pipeline consists of feature engineering steps and the MLPRegressor model configured with the identified hyperparameters.

The final model was trained on the training dataset (testing set represents 20% of the data) and evaluated on both the training and testing datasets. The performance of the final model is as follows:

- Final Train Score: 0.21626702979094825
- Final Test Score: 0.21471498848249027

The train score represents the model's performance on the training dataset, while the test score indicates the model's performance on the unseen testing dataset. These scores reflect the accuracy of the model in predicting the number of steps (n_steps) for recipes based on the given features.

**Model Performance Analysis**

Comparing the final model's performance to the baseline model, we observe an enourmous improvement in the predictive capabilities. The baseline model had much lower scores (Train Score: 0.027, Test Score: 0.020) compared to the final model. This indicates that the final model captures more variability in the target variable (n_steps) based on the selected features.

The enhancements in the final model's performance can be attributed to the following factors:

- Feature Engineering: The additional feature engineering steps, such as binarizing the 'minutes' feature and standardizing the 'calories' feature, helped in transforming and normalizing the data. These transformations made the features more suitable for the modeling process, potentially capturing important patterns and relationships in the data.

- Optimized Hyperparameters: The grid search allowed us to systematically explore different combinations of hyperparameters. By identifying the best hyperparameters, we optimized the model's configuration, enabling it to better capture the underlying patterns in the data.

By incorporating these improvements, the final model demonstrates better accuracy in predicting the number of steps for recipes based on the given features.

**Conclusion**

In conclusion, we developed a final model using an MLPRegressor algorithm with optimized hyperparameters obtained through grid search. The model incorporated feature engineering steps to preprocess the data and improve its performance. The final model outperformed the baseline model in terms of accuracy, showcasing its enhanced predictive capabilities.

Although our final model exhibited substantial improvement compared to the baseline model, we recognize that its accuracy implies the potential existence of other algorithms or features that could yield even higher scores. In terms of future directions, we aim to explore the incorporation of additional features such as tags and ingredients. Unfortunately, due to limitations in computational resources, we were unable to encode and analyze these features in our current model. However, we acknowledge that incorporating such information may provide valuable insights and potentially enhance the model's predictive performance.

---

## Fairness Analysis

In order to analyze the fairness of our final model, we performed a permutation test on our data. This test was run with 1000 trials at a significance of 0.05. The following hypotheses were used to lead this test:

- **Null Hypothesis**: The model's regression coefficient will be the same between high average ratin and low average rating, and any difference is due to random chance
- **Alternative Hypothesis**: The model's score is biased and its score depends on whether or not the recipe has a high or low average rating

In order to perform this permutation test, we split the data into two groups: recipes with a high rating (defined as having an average rating greater than 3.5) and recipes with a low rating (average rating of 3.5 or lower). We then used these groups to calculate regression coefficients on the final model (we used the regression coefficient as opposed to the RMSE for the same reason stated under framing the problem), and subtracted the regression coefficient. The resulting value is our observed statistic. 

The test statistics will be calculated in a similar manner:

- **Test Statistic**: Difference in Regression Coefficients based on Average Rating
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Regression Coefficients of Recipes with High Average Rating</mtext>
  <mo>&#x2212;<!-- − --></mo>
  <mtext>Regression Coefficients of Recipes with Low Average Rating</mtext>
</math> 

 The test statistics found by this permutation test are graphed below, with the red line representing the observed test statistic:


<iframe src="permutation.html" width=800 height=600 frameBorder=0></iframe>

The p-value is calculated to be 0.101, which fails to reject the null hypothesis at a significance of 0.05.


**Conclusion**

Since the permutation test failed to reject the null hypothesis, it is likely (although not definitive) that our model is fair.

---