# Въпроси за R^2 (adj)

## What is the problem with always using R^2?
 - Когато се добавят нови характеристики към модела, **R^2 винаги расте или остава със същата стойност**, дори когато тези нови характеристики **не подобряват модела**
 - Това означава, че R^2 **не наказва** модела за добавяне на безполезни характеристики и може да създаде **грешно впечатление** за подобрение

## How does using R^2 (adj) help solve this problem?
 - Ако се добави някаква нова характеристика към модела, **R^2 (adj) може да намалее**, ако новата харектеристика **не подобрява модела**
 - Това е така, защото **броят на харектеристиките** участва в изчислението на R^2 (adj) и **множество безполезни характеристики** биха **намалили стойността** на метриката
 - **R^2 (adj)** е подходяща метрика за правене на **избор кои харектеристики** да бъдат включени в модела
 - Изборът на правилните характеристики **помага за предотвратяване на overfitting**

## How could we calculate R^2 (adj) in Python?
 - В `sklearn` е налична само **R^2** метриката, но не и **R^2 (adj)**
 - Трябва да изчислим R^2 (adj) по следния начин:
```python
def r2_adjusted_score(samples_count, features_count, r2_score):
    return 1 - (1 - r2_score) * (samples_count - 1) / (samples_count - features_count - 1)
```

## Източници на информация:
 - [Adjusted R-Squared: A Clear Explanation with Examples](https://www.datacamp.com/tutorial/adjusted-r-squared)
 - [How to calculated the adjusted R2 value using scikit](https://stackoverflow.com/questions/51038820/how-to-calculated-the-adjusted-r2-value-using-scikit)