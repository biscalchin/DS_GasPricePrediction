from data_scraper import *  # Import functions from data_scraper.py
from linear_Regression import *
from polinomial_Regression import *
from random_forest import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from artificial_neural_network import *
import time


def calculate_accuracy(mse):
    try:
        accuracy = (1 - mse) * 100  # as per your script
        return round(accuracy, 2)
    except Exception as e:
        print("Error in calculating accuracy: ", e)
        return None


def print_section_header(title):
    print("\n" + "=" * 75)
    print(f" {title} ".center(75))
    print("=" * 75)


def forecaster():
    try:
        spinner_tree = Spinner()
        spinner_forest = Spinner()
        # Load data using the load_data function
        print("Loading Data...")
        data = load_data()
        data = data.fillna(data.mean())

        print(data)

        # Perform feature scaling on the data
        print("Performing Feature Scaling...")
        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        """
        START LINEAR REGRESSION
        """

        start_time_linear = time.time()
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
        stop_time_linear = time.time()
        execution_time_linear = stop_time_linear - start_time_linear

        """
        END LINEAR REGRESSION
        """

        """
        START POLYNOMIAL REGRESSION
        """

        start_time_poly = time.time()
        # Perform polynomial regression with a degree of 16 and get the coefficients
        print("Performing Polynomial Regression...")

        degree = 16  # degree of the polynomial
        coefficients = polynomial_regression(train_data, degree)

        # Calculate and print the Mean Squared Error on the test set for polynomial regression
        print("Calculating Mean Squared Error for Polynomial Regression on Test Set...")

        mse_polynomial = calculate_polynomial_mse(test_data, coefficients)
        stop_time_poly = time.time()
        execution_time_poly = stop_time_poly - start_time_poly

        """
        END POLYNOMIAL REGRESSION
        """

        """
        START DECISION TREE REGRESSION
        """

        start_time_tree = time.time()
        print("\nPerforming Decision Tree Regression...")
        spinner_tree.start()

        # Plot both linear and polynomial regression models along with the data
        # print("Plotting Linear and Polynomial Regression Models along with the Data...")
        # plot_combined_regression(train_data, test_data, coefficients, m, q)

        # Decision Tree Regression
        # Separation of features and target
        X = data.drop('Close', axis=1).values  # Assumi che 'Close' sia il target
        y = data['Close'].values

        # Division into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creation and training of the decision tree regression model
        tree_regressor = DecisionTreeRegressor(min_samples_split=2, max_depth=3)  # Editable parameters (2,5) is optimal
        tree_regressor.fit(X_train, y_train)

        # Predizioni sui dati di test
        y_pred_tree = tree_regressor.predict(X_test)

        # Calcolo dell'errore quadratico medio (MSE)
        mse_tree = mean_squared_error(y_test, y_pred_tree)
        spinner_tree.stop()
        print("Tree completed.")
        stop_time_tree = time.time()
        execution_time_tree = stop_time_tree - start_time_tree

        """
        END DECISION TREE REGRESSION
        """

        """
        START RANDOM FOREST REGRESSION
        """

        start_time_forest = time.time()
        print("\n\nPerforming Random Forest Regression...")
        spinner_forest.start()
        # Creazione e addestramento del modello della foresta casuale
        random_forest = RandomForestRegressor(n_estimators=8, min_samples_split=2, max_depth=3)
        random_forest.fit(X_train, y_train)

        # Predizioni e valutazione del modello
        y_pred_forest = random_forest.predict(X_test)
        mse_forest = mean_squared_error(y_test, y_pred_forest)
        spinner_forest.stop()
        print("Forest completed")
        stop_time_forest = time.time()
        execution_time_forest = stop_time_forest - start_time_forest
        """
        END RANDOM FOREST REGRESSION
        """

        graph_spinner = Spinner()
        print("Plotting results...")
        graph_spinner.start()
        # Chiamata della funzione plot_decision_tree_regression
        plot_combined_regression_with_decision_tree(train_data, test_data, coefficients, m, q, tree_regressor)
        graph_spinner.stop()

        """
        START ARTIFICIAL NEURAL NETWORK
        """
        start_time_ann = time.time()
        try:
            print("\n\nPerforming ANN Regression...")
            data = (data - data.mean()) / data.std()
            # Separazione delle caratteristiche e del target
            X = data.drop('Close', axis=1).values  # Assumendo che 'Close' sia il target
            y = data['Close'].values.reshape(-1, 1)  # Reshape y per adattarlo alle dimensioni attese dall'ANN

            # Divisione in set di addestramento e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Definizione delle dimensioni della rete neurale
            input_size = X_train.shape[1]  # Numero di caratteristiche
            hidden_sizes = [20, 10, 5]  # Dimensioni degli strati nascosti
            output_size = 1  # Output size per la regressione

            # Inizializzazione e addestramento dell'ANN
            ann_model = ImprovedNeuralNetwork([input_size] + hidden_sizes + [output_size], learning_rate=0.0015, epochs=5000)
            ann_model.train(X_train, y_train)

            # Valutazione dell'ANN
            y_pred_ann = ann_model.predict(X_test)
            mse_ann = mean_squared_error(y_test, y_pred_ann)

            # Visualizzazione opzionale
            ann_model.plot_loss()
            ann_model.plot_predictions(X_test, y_test)

        except Exception as e:
            print("\n\nException encountered:", e)
            print("Artificial Neural Network Critical Failure.")
            print("Something went wrong in the data collection process.")
            print("The yFinance API didn't provide valid data for the analysis.")
            mse_ann = -1
        stop_time_ann = time.time()
        execution_time_ann = stop_time_ann - start_time_ann
        """
        END ARTIFICIAL NEURAL NETWORK
        """
        print()
        print_section_header("Mean Square Error (MSE)")
        print("")
        print(f"Linear Regression MSE: {mse_linear}")
        print(f"Polynomial Regression MSE: {mse_polynomial}")
        print(f"Decision Tree MSE: {mse_tree}")
        print(f"Random Forest MSE: {mse_forest}")
        print(f"ANN Regression MSE: {mse_ann}")
        print_section_header("Execution Time")
        print("")
        print(f"Linear Regression Execution Time: {execution_time_linear} seconds")
        print(f"Polynomial Regression Execution Time: {execution_time_poly} seconds")
        print(f"Decision Tree Regression Execution Time: {execution_time_tree} seconds")
        print(f"Random Forest Regression Execution Time: {execution_time_forest} seconds")
        print(f"ANN Regression Execution Time: {execution_time_ann} seconds")

        input("\n\nPress Enter to close the app.\n>")

    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print("Exception encountered:", e)


# Run the forecaster function if the script is executed as the main module
if __name__ == '__main__':
    forecaster()

