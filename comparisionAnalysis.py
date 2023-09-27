import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data for price comparisons
np.random.seed(0)
Websites = ['GeM Website', 'Amazon', 'Flipkart', 'Mesos', 'Snapdeal']
prices = np.random.uniform(10, 50, len(Websites))
# other_platform_prices = gem_prices + np.random.uniform(-5, 5, len(Websites))

price_comparison_data = pd.DataFrame({'Website': Websites, 'Price': prices })

# Create visualizations to convey insights

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


# Scatter plot for price correlation
plt.figure(figsize=(8, 6))
sns.scatterplot(x="GeM Price", y="Other Platform Price", data=price_comparison_data, hue='Website', palette='viridis')
plt.title('Price Correlation Between GeM and Other Platform')
plt.xlabel('GeM Price')
plt.ylabel('Other Platform Price')
plt.tight_layout()
plt.show()

