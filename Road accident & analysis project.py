import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\User\Downloads\Road accident analysis dataset.zip")

# Objective 1: Analyze historical road accident data
# Plot the number of accidents per day of the week
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Day_of_week', order=df['Day_of_week'].value_counts().index)
plt.title('Number of Accidents per Day')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 2: Identify high-risk areas and accident hotspots
# Plot top 10 areas with the most accidents
plt.figure(figsize=(12, 6))
top_areas = df['Area_accident_occured'].value_counts().head(10)
sns.barplot(x=top_areas.index, y=top_areas.values)
plt.xticks(rotation=45)
plt.title("Top 10 High-Risk Areas with Most Accidents")
plt.ylabel("Number of Accidents")
plt.xlabel("Area")
plt.tight_layout()
plt.show()

# Objective 3: Classify accident severity
# Bar chart showing the frequency of each accident severity level
severity_counts = df['Accident_severity'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=severity_counts.index, y=severity_counts.values)
plt.title("Accident Severity Classification")
plt.ylabel("Count")
plt.xlabel("Severity")
plt.show()

# Objective 4: Forecast future accident occurrences
# Extract hour from 'Time' column using known format to avoid warning
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
hourly_counts = df['Hour'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_counts.index, y=hourly_counts.values)
plt.title("Accident Frequency by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.grid(True)
plt.show()

# Objective 5: Provide data-driven recommendations
# Print top accident causes, high-risk time periods, and most affected age groups
print("\nTop Accident Causes:")
print(df['Cause_of_accident'].value_counts().head(5), "\n")

print("Highest Risk Time Periods:")
print(hourly_counts.sort_values(ascending=False).head(5), "\n")

print("Most Affected Age Group:")
print(df['Age_band_of_driver'].value_counts().head(3))
