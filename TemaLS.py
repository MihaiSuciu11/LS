import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def load_and_prepare_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].apply(lambda x: iris.target_names[x])
    return df

def exploratory_data_analysis(df):
    print("Primele 5 randuri ale dataset-ului:")
    print(df.head())
    
    print("\nInformatii despre dataset:")
    print(df.info())

    print("\nStatistici descriptive:")
    print(df.describe())
    
    print("\nDistributia variabilelor:")
    sns.pairplot(df, hue='species')
    plt.show()

def preprocess_data(df):
    # Tratarea valorilor lipsa
    df = df.dropna()
    
    # Eliminarea duplicatelor
    df = df.drop_duplicates()
    
    return df

def split_data(df, test_size=0.2):
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_initial_model(X_train, y_train):
    best_k = 1
    best_accuracy = 0
    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        accuracy = model.score(X_train, y_train)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    print(f"Cel mai bun k: {best_k} cu o acuratete de {best_accuracy}")
    return best_k

def build_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acuratetea pe setul de testare: {accuracy}")

# Main
df = load_and_prepare_data()
exploratory_data_analysis(df)
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)
best_k = train_initial_model(X_train, y_train)
model = build_model(X_train, y_train, best_k)
evaluate_model(model, X_test, y_test)
