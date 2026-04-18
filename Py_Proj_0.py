# ============================================================
# PROJECT: AIR QUALITY ANALYSIS AND AQI PREDICTION
# ============================================================

# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import pandas as pd              # data handling
import numpy as np               # numerical operations
import matplotlib.pyplot as plt  # basic visualization
import seaborn as sns            # advanced visualization

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Load dataset from CSV file
df = pd.read_csv("C:\Users\HP\OneDrive\Desktop\Project\Project Database.csv")

# Display basic information
print("Shape of dataset:", df.shape)
print("Columns:", df.columns)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

# Remove missing values
df.dropna(inplace=True)

# Convert date column into proper datetime format
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print("\nData after cleaning:", df.shape)


# ============================================================
# STEP 3: BASIC DATA ANALYSIS
# ============================================================

print("\nNumber of States:", df['state'].nunique())
print("Number of Cities:", df['city'].nunique())
print("Pollutants:", df['pollutant_id'].unique())


# ============================================================
# PROJECT OBJECTIVES
# ============================================================


# Objective 1: Analyze average pollution level per pollutant
# Objective 2: Compare pollution levels across states
# Objective 3: Identify top polluted cities
# Objective 4: Analyze distribution of pollution values
# Objective 5: Study relationship between min and avg pollution
# Objective 6: Visualize correlation between pollution metrics
# Objective 7: Build Linear Regression model for prediction


# ============================================================
# OBJECTIVE 1: AVERAGE POLLUTION PER POLLUTANT
# ============================================================

pollutant_avg = df.groupby('pollutant_id')['pollutant_avg'].mean()

plt.figure(figsize=(10,5))

pollutant_avg.plot(kind='bar', color='royalblue', edgecolor='black')

plt.title("Average Pollution Level per Pollutant (India)",fontsize=14, fontweight='bold')

plt.xlabel("Pollutant Type", fontsize=12)
plt.ylabel("Average Concentration Level", fontsize=12)

plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\nInsight:")
print("PM10 shows the highest average pollution among all pollutants in India.")


# ============================================================
# OBJECTIVE 2: STATE-WISE POLLUTION ANALYSIS
# ============================================================

state_pollution = df.groupby('state')['pollutant_avg'].mean()

plt.figure(figsize=(12,7))

state_pollution.sort_values().plot(kind='barh',color='teal',edgecolor='black')

plt.title("State-wise Average Pollution Levels",fontsize=14, fontweight='bold')

plt.xlabel("Average Pollution Level", fontsize=12)
plt.ylabel("States", fontsize=12)

plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\nInsight:")
print("Delhi shows the highest pollution among all states, indicating severe air quality issues.")

# ============================================================
# OBJECTIVE 3: TOP 10 POLLUTED CITIES
# ============================================================

city_pollution = df.groupby('city')['pollutant_avg'].mean().nlargest(10)

plt.figure(figsize=(12,6))

city_pollution.plot(kind='bar',color='crimson',edgecolor='black')

plt.title("Top 10 Most Polluted Cities",fontsize=14, fontweight='bold')

plt.xlabel("City", fontsize=12)
plt.ylabel("Average Pollution Level", fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\nInsight:")
print("Top cities like Barbil and Angul have extremely high pollution levels and act as pollution hotspots.")

# ============================================================
# OBJECTIVE 4: POLLUTION DISTRIBUTION (HISTOGRAM)
# ============================================================

plt.figure(figsize=(10,5))

plt.hist(df['pollutant_avg'],bins=25,color='skyblue',edgecolor='black')

plt.title("Distribution of Pollution Levels",fontsize=14, fontweight='bold')

plt.xlabel("Pollution Level", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\nInsight:")
print("Most pollution values are concentrated in lower ranges, but some extreme values indicate severe conditions.")

# ============================================================
# OBJECTIVE 5: SCATTER PLOT (RELATIONSHIP)
# ============================================================

plt.figure(figsize=(8,6))

sns.scatterplot(x='pollutant_min',y='pollutant_avg',data=df,color='purple')

plt.title("Relationship Between Minimum and Average Pollution",fontsize=14, fontweight='bold')

plt.xlabel("Minimum Pollution Level", fontsize=12)
plt.ylabel("Average Pollution Level", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\nInsight:")
print("There is a positive relationship between minimum and average pollution levels.")

# ============================================================
# OBJECTIVE 6: CORRELATION HEATMAP
# ============================================================

corr = df[['pollutant_min','pollutant_max','pollutant_avg']].corr()

plt.figure(figsize=(7,5))

sns.heatmap(corr,annot=True,cmap='coolwarm',linewidths=0.5,fmt=".2f")

plt.title("Correlation Between Pollution Metrics",fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
print("\nInsight:")
print("Strong correlation exists between pollutant_min, pollutant_max, and pollutant_avg.")

# ============================================================
# OBJECTIVE 7: MACHINE LEARNING (LINEAR REGRESSION)
# ============================================================

# Filter PM2.5 data
pm25 = df[df['pollutant_id'] == 'PM2.5']

# Define features and target
X = pm25[['pollutant_min']]
y = pm25['pollutant_avg']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# ============================================================
# LINEAR REGRESSION PLOT
# ============================================================

plt.figure(figsize=(8,6))

# Actual values
plt.scatter(X_test, y_test,color='blue',alpha=0.6,label='Actual Data')

# Regression line
plt.plot(X_test, y_pred,color='red',linewidth=2,label='Regression Line')

plt.title("Linear Regression: PM2.5 Min vs Average",fontsize=14, fontweight='bold')

plt.xlabel("PM2.5 Minimum Value", fontsize=12)
plt.ylabel("PM2.5 Average Value", fontsize=12)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\nInsight:")
print("Linear Regression model shows moderate accuracy, indicating that pollution prediction is possible but can be improved with more features.")



# ============================================================
# BOXPLOT (POLLUTION SPREAD)
# ============================================================

plt.figure(figsize=(8,5))

sns.boxplot(x=df['pollutant_avg'], color='orange')


plt.title("Boxplot of Pollution Levels (Spread & Outliers)",
          fontsize=14, fontweight='bold')

plt.xlabel("Pollution Level", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\nInsight:")
print("Boxplot shows presence of outliers, indicating extreme pollution levels in some regions.")


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n================ FINAL CONCLUSION ================")

print("""
1. PM10 and PM2.5 are the major contributors to air pollution.

2. Certain states like Delhi show extremely high pollution levels.

3. Several cities act as pollution hotspots and require attention.

4. Pollution data shows skewed distribution with some extreme values.

5. Strong relationships exist between different pollution metrics.

6. Machine Learning model provides a basic prediction capability.

7. Overall, air pollution in India is a serious concern and requires data-driven solutions.
""")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n✅ PROJECT COMPLETED SUCCESSFULLY")
