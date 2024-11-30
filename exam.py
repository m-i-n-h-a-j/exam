*Support vector machine(SVM)*
import numpy as np                                                     from sklearn.model_selection import train_test_split                   from sklearn.preprocessing import StandardScaler                       from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.svm import SVC                                                                                                                   # Load the dataset (Iris dataset used for demonstration purposes)      data = load_iris()                                                     X = data.data                                                          y = data.target                                                        
# Split data into training and testing sets                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)                                                     
# Standardizing the data                                               scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)                                X_test = scaler.transform(X_test)                                      
# Initialize and train Support Vector Machine classifier               svm = SVC()
svm.fit(X_train, y_train)                                              
# Predict using Support Vector Machine                                 y_pred = svm.predict(X_test)                                                                                                                  # Print accuracy and classification report
print("--- Support Vector Machine ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")                   print(f"Classification Report:\n{classification_report(y_test, y_pred)}")                                                                                                                                            -----------------------                                                                                                                       *Decision tree*
import numpy as np                                                     from sklearn.model_selection import train_test_split                   from sklearn.preprocessing import StandardScaler                       from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris                                 from sklearn.tree import DecisionTreeClassifier
                                                                       # Load the dataset (Iris dataset used for demonstration purposes)      data = load_iris()                                                     X = data.data                                                          y = data.target                                                        
# Split data into training and testing sets                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)                                                                                                                            # Standardizing the data                                               scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)                                                                                                             # Initialize and train Decision Tree classifier                        decision_tree = DecisionTreeClassifier()                               decision_tree.fit(X_train, y_train)
                                                                       # Predict using Decision Tree                                          y_pred = decision_tree.predict(X_test)                                 
# Print accuracy and classification report                             print("--- Decision Tree ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")                   print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
                                                                       --------------------
                                                                       *Logistic Regeression*
import numpy as np                                                     from sklearn.model_selection import train_test_split                   from sklearn.preprocessing import StandardScaler                       from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression                                                                                           # Load the dataset (Iris dataset used for demonstration purposes)
data = load_iris()                                                     X = data.data                                                          y = data.target                                                        
# Split data into training and testing sets                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)                                                     
# Standardizing the data                                               scaler = StandardScaler()                                              X_train = scaler.fit_transform(X_train)                                X_test = scaler.transform(X_test)

# Initialize and train Logistic Regression classifier
logistic_regression = LogisticRegression()                             logistic_regression.fit(X_train, y_train)                                                                                                     # Predict using Logistic Regression                                    y_pred = logistic_regression.predict(X_test)                                                                                                  # Print accuracy and classification report
print("--- Logistic Regression ---")                                   print(f"Accuracy: {accuracy_score(y_test, y_pred)}")                   print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
                                                                       --------------
                                                                       *k-means clustering*                                                   import numpy as np
from sklearn.model_selection import train_test_split                   from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris                                 from sklearn.cluster import KMeans
                                                                       # Load the dataset (Iris dataset used for demonstration purposes)
data = load_iris()                                                     X = data.data                                                          y = data.target                                                        
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)                                                     
# Standardizing the data                                               scaler = StandardScaler()                                              X_train = scaler.fit_transform(X_train)                                X_test = scaler.transform(X_test)
                                                                       # K-Means Clustering (Unsupervised Learning)                           kmeans = KMeans(n_clusters=3, random_state=42)                         kmeans.fit(X_train)                                                    
# Predict clusters for test data
y_kmeans_pred = kmeans.predict(X_test)

# Print cluster centers and predicted clusters                         print("--- K-Means Clustering ---")                                    print(f"Cluster centers: {kmeans.cluster_centers_}")                   print(f"Predicted clusters for test data: {y_kmeans_pred}")            
----------------                                                                                                                              *Mean removal*                                                         # Mean Removal or Standardization
import numpy as np                                                     from sklearn import preprocessing
                                                                       # Sample data                                                          input_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],                                                      [0.5, -7.9, 5.6],
                       [5.9, 2.3, -5.8]])                              
# Mean and Standard Deviation before Standardization                   print("\nMean before standardization =", input_data.mean(axis=0))
print("Std deviation before standardization =", input_data.std(axis=0))                                                                       # Apply Standardization                                                data_scaled = preprocessing.scale(input_data)
print("Mean after standardization =", data_scaled.mean(axis=0))
print("Std deviation after standardization =", data_scaled.std(axis=0))                                                                       --------------                                                                                                                                                                                                       *K nearest neighbours*                                                 import numpy as np
from sklearn.model_selection import train_test_split                   from sklearn.preprocessing import StandardScaler                       from sklearn.metrics import accuracy_score, classification_report      from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier                     
# Load the dataset (Iris dataset used for demonstration purposes)      data = load_iris()                                                     X = data.data
y = data.target                                                        
# Split data into training and testing sets                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)                                                                                                                            # Standardizing the data                                               scaler = StandardScaler()                                              X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)                                                                                                             # Initialize K-Nearest Neighbors classifier
knn = KNeighborsClassifier()
                                                                       # Train the classifier                                                 knn.fit(X_train, y_train)                                              
# Predict the labels for the test data                                 y_pred = knn.predict(X_test)                                           
# Print accuracy and classification report                             print("--- K-Nearest Neighbors ---")                                   print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
                                                                       --------------                                                         
*Min max scaling*                                                      # Scaling using Min-Max Scaler                                         import numpy as np                                                     from sklearn import preprocessing
                                                                       # Sample data                                                          input_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],
                       [0.5, -7.9, 5.6],                                                      [5.9, 2.3, -5.8]])
                                                                       # Apply Min-Max Scaling                                                data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))  data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)      print("\nMin-max scaled data:\n", data_scaled_minmax)                                                                                         -----------------                                                                                                                                                                                                    *Naive bayes*                                                          import numpy as np                                                     from sklearn.model_selection import train_test_split                   from sklearn.preprocessing import StandardScaler                       from sklearn.metrics import accuracy_score, classification_report      from sklearn.datasets import load_iris                                 from sklearn.naive_bayes import GaussianNB                                                                                                    # Load the dataset (Iris dataset used for demonstration purposes)      data = load_iris()                                                     X = data.data                                                          y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Naive Bayes classifier
naive_bayes = GaussianNB()

# Train classifier
naive_bayes.fit(X_train, y_train)

# Predict
y_pred = naive_bayes.predict(X_test)

# Print accuracy and classification report
print("--- Naive Bayes ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

----------

*Random forest*
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (Iris dataset used for demonstration purposes)
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Random Forest classifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Predict using Random Forest
y_pred = random_forest.predict(X_test)

# Print accuracy and classification report
print("--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

-------------

*Normalization*
# Normalization (L1 and L2)
import numpy as np
from sklearn import preprocessing

# Sample data
input_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],
                       [0.5, -7.9, 5.6],
                       [5.9, 2.3, -5.8]])

# L1 Normalization
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
print("\nL1 normalized data:\n", data_normalized_l1)

# L2 Normalization
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL2 normalized data:\n", data_normalized_l2)

-------------


*Binarization*
# Binarization
import numpy as np
from sklearn import preprocessing

# Sample data
input_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],
                       [0.5, -7.9, 5.6],
                       [5.9, 2.3, -5.8]])

# Apply Binarization
data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input_data)
print("\nBinarized data:\n", data_binarized)


----------
----------
----------

def display_board(board):
print("\n")
print(" " + board[0] + " | " + board[1] + " | " + board[2])
print("---|---|---")
print(" " + board[3] + " | " + board[4] + " | " + board[5])
print("---|---|---")
print(" " + board[6] + " | " + board[7] + " | " + board[8])
print("\n")
# Function to check for a win
def check_win(board, mark):
win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), # Horizontal
(0, 3, 6), (1, 4, 7), (2, 5, 8), # Vertical
(0, 4, 8), (2, 4, 6)] # Diagonal
return any(board[a] == board[b] == board[c] == mark for a, b, c in win_conditions)
# Function to check for a draw (no empty spaces)
def check_draw(board):
return ' ' not in board
# Function to handle a player's move
def player_move(board, player):
while True:
try:
move = int(input(f"Player {player}, choose a position (1-9): ")) - 1
if board[move] == ' ':
board[move] = player
break
else:
print("This position is already taken. Try again.")
except (ValueError, IndexError):
print("Invalid input. Please choose a valid position (1-9).")
# Main game function
def tic_tac_toe():
# Initialize the board (a list of 9 spaces)
board = [' '] * 9
current_player = 'X'
# Game loop
while True:
display_board(board)
# Player makes a move
player_move(board, current_player)
# Check if the current player wins
if check_win(board, current_player):
display_board(board)
print(f"Player {current_player} wins!")
break
# Check if the game is a draw
if check_draw(board):
display_board(board)
print("It's a draw!")
break
# Switch to the other player
current_player = 'O' if current_player == 'X' else 'X'
# Start the game
tic_tac_toe()

----------------

def last_coin_standing(n, k):
"""
Interactive Last Coin Standing game.
:param n: Total number of coins
:param k: Maximum number of coins a player can take in one turn
:return: The winner ('Player 1' or 'Player 2')
"""
# Player 1 starts first
current_player = 1
while n > 0:
# Print the current state
print(f"\nCoins left: {n}")
# Player chooses how many coins to take (between 1 and k)
while True:
try:
take = int(input(f"Player {current_player}, choose how many coins to take (1-{min(k,
n)}): "))
if 1 <= take <= min(k, n):
break
else:
print(f"Invalid input. You must take between 1 and {min(k, n)} coins.")
except ValueError:
print("Invalid input. Please enter a number.")
# Reduce the number of coins left
n -= take
# Check if the game is over (i.e., no coins are left)
if n == 0:
print(f"\nPlayer {current_player} takes the last coin and loses!")
return f"Player {1 if current_player == 2 else 2} wins!"
# Switch to the next player
current_player = 1 if current_player == 2 else 2
# Example usage:
n = 10 # Total number of coins
k = 3 # Maximum number of coins that can be taken in a turn
result = last_coin_standing(n, k)
print(result)
