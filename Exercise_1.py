import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

square_feet = np.array([100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600, 634, 718, 750, 850, 903, 978, 1010, 1050, 1990]).reshape(-1, 1)
price = np.array([12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989])

lr = LinearRegression()
lr.fit(square_feet, price)
line_x = np.arange(0, 2000).reshape(-1, 1)
line_y_lr = lr.predict(line_x)

# RANSAC Regression model
ransac = RANSACRegressor()
ransac.fit(square_feet, price)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_y_ransac = ransac.predict(line_x)

plt.figure(figsize=(10,6))
plt.scatter(square_feet[inlier_mask], price[inlier_mask], color='green', marker='.', label='Inliers')
plt.scatter(square_feet[outlier_mask], price[outlier_mask], color='red', marker='.', label='Outliers')
plt.plot(line_x, line_y_lr, color='blue', linestyle='-', linewidth=2, label='Before RANSAC')
plt.plot(line_x, line_y_ransac, color='gold', linestyle='-', linewidth=2, label='after RANSAC')
plt.legend(loc='lower right')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.title('Linear Regression Before and After RANSAC')
plt.show()

print(f"Linear Regression line parameters: Slope = {lr.coef_[0]}, Intercept = {lr.intercept_}")
print(f"RANSAC Regression line parameters: Slope = {ransac.estimator_.coef_[0]}, Intercept = {ransac.estimator_.intercept_}")
