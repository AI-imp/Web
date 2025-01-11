## Government affairs retrieval system
### 1.Project configuration
- MongoDB (text database, good scalability, easy to operate) 
- flask framework (lightweight, easy to operate) 
- test server deployment 
- hybrid recommendation algorithm based on user behavior (ratings, browsing logs)
- data analysis
![img1](https://github.com/AI-imp/Web/blob/main/picture/data.png?raw=true)
## 2.algorithm
### 2.1.One round of search: sort by time and region
### 2.2.Second round of search
#### 2.2.1. Store the policy ID of the returned round of results form, especially the policies that the user has not browsed, into the to-be-recommended list
#### 2.2.2. Import the rating information of this user and other active users
#### 2.2.3. Conduct training to classify groups similar to the user, and predict the user's policy score for the recommended list
#### 2.2.4. Use the score as part of the weight, combine it with a round of retrieval and interest similarity, and then sort and output it.
![img1](https://github.com/AI-imp/Web/blob/main/picture/alg1.png?raw=true)
![img1](https://github.com/AI-imp/Web/blob/main/picture/alg2.png?raw=true)
## 3.Training and Evaluate
```
python3 Model/Model.py
```
![img1](https://github.com/AI-imp/Web/blob/main/picture/moedl.png?raw=true)
## 4.Server deployment
```
python3 server.py
```
![img1](https://github.com/AI-imp/Web/blob/main/picture/view.jpg?raw=true)


