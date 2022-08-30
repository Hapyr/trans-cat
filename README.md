# Wellcome to TransCat!
Categorize your bank transactions automatically according to the logistic regression classification!
## Get started
The project is a simple Flask dockerized web application. So to start just run `docker-compose up`.
The page is then accessible via 127.0.0.1:1234.
The application is intended for local hosting and not suitable for remote access.
## How it works
The main idea of the application is to assign exactly one category to each bank account transaction (e.g. eating, shopping, etc.). 
The import and export format is csv. 
The application adds a "Category" column to the transaction list of your bank.
When using the tool for the first time, you need to assign a category to each transaction itself. With this information, your later transaction lists will be automatically categorized. If there is an error in the categorization, you can change it. If new data is classified, it will be added to the model. 
You can have several models for different banks.

## Impressions
![Bildschirmfoto vom 2022-08-29 23-05-59](https://user-images.githubusercontent.com/23398802/187299643-316d4da1-8b0c-42c4-afb1-dc39b09453ec.png)
![Bildschirmfoto vom 2022-08-29 23-07-27](https://user-images.githubusercontent.com/23398802/187299648-0ab07b0f-3fc0-414a-9184-de01afed58a5.png)
