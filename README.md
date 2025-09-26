# 🍽️ Zomato Restaurant Insights Engine

> A Product Analyst project that predicts restaurant churn risk and recommends hidden gems to users — built with Python, scikit-learn, and Gradio.

## 🚀 Live Demo  
[**Launch on Hugging Face**](https://huggingface.co/spaces/DeekshithaBapathu/zomato_restaurant_insights)

Features:
-  **Restaurant Health Check**: See your risk score + growth tips
-  **Get Recommendations**: Find similar hidden gems
-  **Explore Data**: Filter by location, cost, and rating

##  Problem Statement
- **Users** see the same popular restaurants repeatedly and miss high-quality hidden gems.
- **Restaurants**, especially small/local ones, struggle with low visibility and order volume — often without understanding why.
- **Zomato** loses potential GMV and engagement due to poor discovery and restaurant churn.

This project solves both sides of the marketplace with data-driven insights.

##  Solution Overview
I built a two-part system:
1. **Restaurant Health Score (RHS) Predictor**  
   - ML model (Random Forest) that flags at-risk restaurants using features like rating, delivery time, and order volume.
   - Provides actionable tips: *“Improve delivery by 10 mins”*, *“Add trending dishes”*, etc.

2. **Personalized Recommendation Engine**  
   - Content-based recommender using cuisine, location, and restaurant type.
   - Helps users discover similar hidden gems they’ll love.

 **Validated with simulated A/B testing**:  
- **14.8% increase in orders** for users who saw recommendations  
- **50% more restaurant discovery** → higher platform engagement

##  Key Results
| Metric | Result |
|--------|--------|
| Model Accuracy | 96% |
| AUC Score | 0.99 |
| Simulated Order Uplift | +14.8% |
| Discovery Boost | +50% |
| Deployment | Live on Hugging Face |


