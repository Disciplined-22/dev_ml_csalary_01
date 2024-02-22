import pandas as pd

# Replace 'file_path' with the actual path to your CSV file
file_path = "candidate_information.csv"

# Read the CSV file using pandas
try:
    df = pd.read_csv(file_path)
    print(df)
    print("success")
    print(df.head())
    print("File read successfully!")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print("An error occurred:", e)


education_mapping = {
   '10th grade':0,
     'High school':1,
     'Degree':2,
     "Master's degree":3,
   'Ph.D.':4
}

# Use the replace() function to apply the mapping to the 'Candidate Education' column
df['Candidate Education'] = df['Candidate Education'].replace(education_mapping)

# Now the 'Candidate Education' column contains the corresponding labels
print(df)

X = df[['Candidate Skills', 'Candidate Experience', 'Candidate Education']]

print(X.head())

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Candidate Skills' column
X['Candidate Skills'] = label_encoder.fit_transform(X['Candidate Skills'])

print(X.head())


y = df[['Candidate Salary']]
print(y)

# Define weights for each feature
Weight_Skills = 0.5
Weight_Experience = 0.3
Weight_Education = 0.2

# Assuming you have candidate data stored in variables
# Replace candidate_skills, candidate_experience, and candidate_education with your actual data
candidate_skills = 4.5
candidate_experience = 10
candidate_education = 2

# Calculate Candidate Score
# Candidate_Score = (Weight_Skills * candidate_skills) + (Weight_Experience * candidate_experience) + (Weight_Education * candidate_education)

import pandas as pd

# Assuming you have DataFrame df with columns 'Candidate Skills', 'Candidate Experience', and 'Candidate Education'
# Assuming Weight_Skills, Weight_Experience, and Weight_Education are defined

# Calculate Candidate Score
df['Candidate Score'] = (Weight_Skills * df['Candidate Skills']) + (Weight_Experience * df['Candidate Experience']) + (Weight_Education * df['Candidate Education'])

# Calculate Pearson's correlation coefficient between Candidate Score and Candidate Salary
correlation = df['Candidate Score'].corr(df['Candidate Salary'])

# Print the correlation coefficient
print(f"Pearson's correlation coefficient: {correlation:.2f}")


from sklearn.model_selection import train_test_split

# Assuming X and y are your features and target variable respectively
# Replace X and y with your actual data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a RandomForestRegressor model
model = RandomForestRegressor()

# Fit the model to your training data
model.fit(X_train, y_train)

import joblib

# Predict using the trained model
predictions = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
joblib.dump(model, 'model.pkl')
