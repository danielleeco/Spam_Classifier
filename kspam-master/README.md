# kspam

Курсовая работа по классификации спама.

Используемый датасет - https://www.kaggle.com/uciml/sms-spam-collection-dataset

## Usage
```
# установка зависимостей (увы, не всех, надо дополнить)
pip install -r requirements.txt

## необходимо заранее обучить катбуст (так как долговато) и сохранить модель под именем cb.model.bin
## это можно сделать так
# python train_catboost.py

# запуск streamlit-приложения
streamlit run app.py
```


## Results

| model       | accuracy score     |
| ----------- | ------------------ |
| catboost    | 0.9853181076672104 |
| logreg      | 0.9787928221859706 |
| naive bayes | 0.9575856443719413 |


## Screenshots
<details>
<summary>only ham</summary>

![](https://user-images.githubusercontent.com/54478880/80927261-2fd79800-8da5-11ea-8880-113fdb8e20b0.png)
</details>


<details>
<summary>1/3 ham</summary>

![](https://user-images.githubusercontent.com/54478880/80927247-24846c80-8da5-11ea-94c5-b6bfd6815e0e.png)
</details>
