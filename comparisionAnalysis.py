import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import ttest_ind



####   Regression Analysis for Price Change Prediction

# Generate synthetic data for product prices over time
np.random.seed(0)
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
price_data_for_pridiction = np.random.uniform(10, 50, size=len(date_rng))


# Generate synthetic data for price comparisons
np.random.seed(0)
Websites = ['GeM Website', 'Amazon', 'Flipkart', 'Mesos', 'Snapdeal']
prices = np.random.uniform(10, 50, len(Websites))




def comparision_analysis():
    price_comparison_data = pd.DataFrame({'Website': Websites, 'Price': prices })
    # Bar chart for price comparison
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.barplot(x="Website", y="Price", data=price_comparison_data, color='skyblue', label='Price')
    # sns.barplot(x="Website", y="Other Platform Price", data=price_comparison_data, color='lightcoral', label='Other Platform Price')
    plt.title('Price Comparison Between GeM and Other Platform')
    plt.xlabel('Website')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def price_pridiction():
    product_prices_for_pridiction = pd.DataFrame({'Date': date_rng, 'Price': price_data_for_pridiction})
    # Create a lag feature for the previous day's price
    product_prices_for_pridiction['Previous Price'] = product_prices_for_pridiction['Price'].shift(1)

    # Drop the first row with NaN values
    product_prices_for_pridiction = product_prices_for_pridiction.dropna()

    # Split the data into training and testing sets
    X = product_prices_for_pridiction[['Previous Price']]
    y = product_prices_for_pridiction['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE) to evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')

    # Predict future price change
    future_price = product_prices_for_pridiction.iloc[-1]['Price']
    future_previous_price = product_prices_for_pridiction.iloc[-1]['Previous Price']
    predicted_future_price_change = model.predict(np.array([[future_previous_price]]))

    print(f'Regression Analysis Predicted Future Price Change: {predicted_future_price_change[0]:.2f}')

def review_analysis():
    
    # Number of recent users (reviews)
    n_users = 10

    # Generate data for GeM reviews
    gem_reviews = {
        'User': [f'User {i}' for i in range(1, n_users + 1)],
        'GeM Stars': np.random.randint(1, 6, n_users)  # Ratings from 1 to 5 stars
    }

    # Generate data for Amazon reviews
    amazon_reviews = {
        'User': [f'User {i}' for i in range(1, n_users + 1)],
        'Amazon Stars': np.random.randint(1, 6, n_users)
    }

    # Generate data for another website reviews
    other_website_reviews = {
        'User': [f'User {i}' for i in range(1, n_users + 1)],
        'Other Website Stars': np.random.randint(1, 6, n_users)
    }

    # Create DataFrames from the generated data
    gem_df = pd.DataFrame(gem_reviews)
    amazon_df = pd.DataFrame(amazon_reviews)
    other_website_df = pd.DataFrame(other_website_reviews)

    # Merge the DataFrames on the 'User' column for comparison
    merged_df = gem_df.merge(amazon_df, on='User').merge(other_website_df, on='User')

    # Plot star ratings comparison
    plt.figure(figsize=(10, 6))
    plt.bar(merged_df['User'], merged_df['GeM Stars'], label='GeM', color='skyblue')
    plt.bar(merged_df['User'], merged_df['Amazon Stars'], label='Amazon', color='lightcoral', alpha=0.7)
    plt.bar(merged_df['User'], merged_df['Other Website Stars'], label='Other Website', color='lightgreen', alpha=0.7)

    plt.xlabel('User')
    plt.ylabel('Star Ratings')
    plt.title('Star Ratings Comparison by User')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # Calculate average ratings on each platform
    avg_gem_rating = gem_df['GeM Stars'].mean()
    avg_amazon_rating = amazon_df['Amazon Stars'].mean()
    avg_other_website_rating = other_website_df['Other Website Stars'].mean()

    # Plot star ratings comparison along with average ratings
    plt.figure(figsize=(12, 6))

    # Average ratings
    plt.axhline(y=avg_gem_rating, color='blue', linestyle='--', label=f'Avg GeM ({avg_gem_rating:.2f})')
    plt.axhline(y=avg_amazon_rating, color='red', linestyle='--', label=f'Avg Amazon ({avg_amazon_rating:.2f})')
    plt.axhline(y=avg_other_website_rating, color='green', linestyle='--', label=f'Avg Other ({avg_other_website_rating:.2f})')

    plt.xlabel('User')
    plt.ylabel('Star Ratings')
    plt.title('Star Ratings Comparison by User')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def trend_analysis():
    product_prices_for_pridiction = pd.DataFrame({'Date': date_rng, 'Price': price_data_for_pridiction})
    # Create a line chart to visualize the product prices over time
    plt.figure(figsize=(12, 6))
    plt.plot(product_prices_for_pridiction['Date'], product_prices_for_pridiction['Price'], marker='o', linestyle='-', label='Price')


    # Fit a linear regression model to identify the linear trend
    X = np.arange(len(product_prices_for_pridiction)).reshape(-1, 1)
    y = product_prices_for_pridiction['Price']
    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)
    plt.plot(product_prices_for_pridiction['Date'], trend_line, color='red', linestyle='--', label='Linear Trend')

    plt.legend()
    plt.tight_layout()
    plt.show()


    # Fit a polynomial regression model (2nd degree)
    degree = 2
    X = np.arange(len(product_prices_for_pridiction)).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, product_prices_for_pridiction['Price'])
    price_poly = model.predict(X_poly)
    plt.plot(product_prices_for_pridiction['Date'], price_poly, color='red', linestyle='--', label=f'Polynomial Regression (Degree {degree})')

    plt.title('Product Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
review_analysis()
comparision_analysis()
price_pridiction()
trend_analysis()

