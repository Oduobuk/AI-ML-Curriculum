# Week 3 Assessment: Anomaly Detection & Recommendation Systems

## Multiple Choice (2 points each)

1. Which of the following is NOT a typical characteristic of anomalies in a dataset?
   a) They are rare compared to normal data points
   b) They have different patterns than normal data
   c) They are always errors that should be removed
   d) They can represent important events

2. In the Isolation Forest algorithm, how are anomalies typically identified?
   a) By measuring the distance to the k-nearest neighbors
   b) By calculating the reconstruction error
   c) By having shorter path lengths in the isolation trees
   d) By having higher density than normal points

3. Which of the following is an advantage of content-based filtering over collaborative filtering?
   a) It can recommend items to users with unique tastes
   b) It doesn't require item features
   c) It performs better with sparse data
   d) It can capture serendipitous recommendations

4. In matrix factorization for recommendation systems, what does the latent factor k represent?
   a) The number of users in the system
   b) The number of items in the system
   c) The number of features used to represent users and items
   d) The learning rate of the algorithm

5. Which evaluation metric is most appropriate for a recommendation system where the order of recommendations matters?
   a) RMSE
   b) MAE
   c) Precision@k
   d) NDCG

## Short Answer (5 points each)

6. Explain the cold start problem in recommendation systems and describe two strategies to mitigate it.

7. Compare and contrast the Isolation Forest and One-Class SVM algorithms for anomaly detection. When would you choose one over the other?

8. What is the difference between user-based and item-based collaborative filtering? Provide an example of when you might prefer one over the other.

## Practical Exercise (10 points)

9. Implement a function that calculates the precision@k and recall@k metrics for a recommendation system. The function should take:
   - A list of recommended items
   - A list of relevant items
   - The value of k
   
   ```python
   def evaluate_recommendations(recommended, relevant, k=5):
       # Your implementation here
       pass
   ```

## Case Study (15 points)

10. You're building a recommendation system for an e-commerce platform that sells both books and electronics. The platform has:
    - User purchase history
    - Product descriptions and categories
    - User ratings (1-5 stars)
    - Browsing history
    
    Design a hybrid recommendation system that:
    a) Combines content-based and collaborative filtering
    b) Addresses the cold start problem for new users and items
    c) Considers the different nature of books vs. electronics in recommendations
    
    Provide pseudocode or a clear explanation of your approach.

## Bonus (5 points)

11. How would you modify a traditional recommendation system to handle the concept of "surprise" or serendipity in recommendations? What metrics would you use to evaluate the success of such a system?
