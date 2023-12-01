from data_scraper import *  # Import functions from data_scraper.py
from linear_Regression import *
from polinomial_Regression import *
from random_forest import *


def calculate_accuracy(mse):
    try:
        accuracy = (1 - mse) * 100  # as per your script
        return round(accuracy, 2)
    except Exception as e:
        print("Error in calculating accuracy: ", e)
        return None


def forecaster():
    try:
        # Load data using the load_data function
        print("Loading Data...")
        data = load_data()
        print(data)

        # Perform feature scaling on the data
        print("Performing Feature Scaling...")
        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        # Split the data into training and testing sets using train_test_split
        print("Splitting Data into Training and Testing Sets...")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Define learning rate and number of iterations for gradient descent
        learning_rate = 0.001
        num_iterations = 1000

        # Perform linear regression using gradient descent and get the slope (m) and intercept (q)
        print("Performing Linear Regression using Gradient Descent...")
        try:
            print("Trying to compute using CUDA GPU technology...")
            m, q = gradient_descent_gpu(train_data, learning_rate, num_iterations)
        except Exception as e:
            print(e)
            print("Unable to compute using parallelization.")
            print("Using CPU computation: Warning! Process will take longer...")
            m, q = gradient_descent(train_data, learning_rate, num_iterations)

        # Calculate and print the Mean Squared Error on the test set for linear regression
        print("Calculating Mean Squared Error for Linear Regression on Test Set...")

        mse_linear = calculate_linear_mse(test_data, m, q)

        print(f"Mean Squared Error on Test Set for the linear regression is: {mse_linear}")

        # Perform polynomial regression with a degree of 16 and get the coefficients
        print("Performing Polynomial Regression...")

        degree = 16  # degree of the polynomial
        coefficients = polynomial_regression(train_data, degree)

        # Calculate and print the Mean Squared Error on the test set for polynomial regression
        print("Calculating Mean Squared Error for Polynomial Regression on Test Set...")

        mse_polynomial = calculate_polynomial_mse(test_data, coefficients)
        print(f"Mean Squared Error on Test Set for the polynomial regression is {mse_polynomial}")

        # For Linear Regression
        accuracy_linear = calculate_accuracy(mse_linear)
        print(f"Linear regression prediction accuracy: {accuracy_linear}%")

        # For Polynomial Regression
        accuracy_polynomial = calculate_accuracy(mse_polynomial)
        print(f"Polynomial regression prediction accuracy: {accuracy_polynomial}%")

        # Plot both linear and polynomial regression models along with the data
        print("Plotting Linear and Polynomial Regression Models along with the Data...")
        plot_combined_regression(train_data, test_data, coefficients, m, q)

        # Decision Tree Regression
        # Separation of features and target
        X = data.drop('Close', axis=1).values  # Assumi che 'Close' sia il target
        y = data['Close'].values

        # Division into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creation and training of the decision tree regression model
        tree_regressor = DecisionTreeRegressor(min_samples_split=2, max_depth=5)  # Editable parameters (2,5) is optimal
        tree_regressor.fit(X_train, y_train)

        # Predizioni sui dati di test
        y_pred = tree_regressor.predict(X_test)

        # Calcolo dell'errore quadratico medio (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Decision Tree MSE: {mse}")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[:, 0], y_test, color='blue', label='Real Data',
                    alpha=0.5)  # Assumi che X_test[:, 0] sia una variabile significativa
        plt.scatter(X_test[:, 0], y_pred, color='red', label='Prediction', alpha=0.25)
        plt.title("Regression with decision Tree")
        plt.xlabel("Feature")
        plt.ylabel("Close")
        plt.legend()
        plt.show()

        # Chiamata della funzione plot_decision_tree_regression
        plot_combined_regression_with_decision_tree(train_data, test_data, coefficients, m, q, tree_regressor)

        # Creazione e addestramento del modello della foresta casuale
        random_forest = RandomForestRegressor(n_estimators=10, min_samples_split=2, max_depth=2)
        random_forest.fit(X_train, y_train)

        # Predizioni e valutazione del modello
        y_pred = random_forest.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Random Forest MSE: {mse}")




    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print("Exception encountered:", e)


# Run the forecaster function if the script is executed as the main module
if __name__ == '__main__':
    forecaster()
