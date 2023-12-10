# Importing necessary libraries and modules
from data_scraper import *  # Importing functions from data_scraper.py for data scraping
from linear_Regression import *  # Importing linear regression-related functions
from polynomial_Regression import *  # Importing polynomial regression-related functions
from random_forest import *  # Importing random forest-related functions
from sklearn.model_selection import train_test_split  # For splitting the data into train and test sets
from sklearn.metrics import mean_squared_error  # For calculating mean squared error of predictions
from sklearn.metrics import roc_curve, auc  # For generating ROC curves and calculating AUC
from artificial_neural_network import *  # Importing ANN-related functions
import time  # For tracking execution time


# Function to print section headers in the console
def print_section_header(title):
    """
    Prints a formatted header for different sections of output.

    Args:
    title (str): Title of the section.

    Returns:
    None
    """
    print("\n" + "=" * 75)
    print(f" {title} ".center(75))  # Center-aligns the title
    print("=" * 75)


# Function to convert regression predictions to binary values based on a threshold
def convert_to_binary(predictions, threshold):
    """
    Converts continuous prediction values into binary (0 or 1) based on a specified threshold.

    Args:
    predictions (list or array): The list of predicted values.
    threshold (float): The threshold for conversion to binary.

    Returns:
    list: A list of binary values.
    """
    return [1 if pred >= threshold else 0 for pred in predictions]


def plot_graph(train_data, test_data, coefficients, m, q, tree_regressor):
    """
    Plots a combined regression graph with an overlaid decision tree regression.

    This function initiates a spinner, signaling the start of the graph plotting process.
    It then calls the 'plot_combined_regression_with_decision_tree' function to generate
    and display the graph. The graph combines linear regression results with a decision
    tree regression overlay. After plotting, the spinner is stopped, indicating the
    completion of the process.

    Parameters:
    - train_data: Dataset used for training the regression model.
    - test_data: Dataset used for testing the regression model.
    - coefficients: Coefficients of the linear regression model.
    - m: The slope of the linear regression line.
    - q: The intercept of the linear regression line.
    - tree_regressor: The decision tree regressor instance.

    Note: This function assumes that all input parameters are pre-processed and
    ready for use in plotting. Ensure that 'train_data' and 'test_data' are scaled
    and transformed appropriately before passing them to this function.

    The function is designed to run in a separate thread to avoid blocking the main
    execution flow while the graph is being plotted.
    """
    graph_spinner = Spinner()
    print("Plotting results...")

    graph_spinner.start()

    # Qui inserisci il tuo codice per generare il grafico
    plot_combined_regression_with_decision_tree(train_data, test_data, coefficients, m, q, tree_regressor)

    graph_spinner.stop()


# Function to generate predictions for polynomial regression
def polynomial_predictions(data, coefficients):
    """
    Calculates predictions based on polynomial regression coefficients.

    Args:
    data (DataFrame): DataFrame containing the feature 'Numerical_Index_scaled'.
    coefficients (list): Coefficients of the polynomial regression.

    Returns:
    np.array: Predicted values.
    """
    X = data['Numerical_Index_scaled'].values
    y_pred = np.zeros_like(X)

    for i, coeff in enumerate(coefficients):
        y_pred += coeff * X ** i

    return y_pred


# Main function to perform various regression analyses
def forecaster():
    """
    Main function to run different regression models, calculate metrics, and plot results.

    Args:
    None

    Returns:
    None
    """
    try:
        # Initialize spinners for UI feedback during long-running processes
        spinner_tree = Spinner()  # For Decision Tree Regression
        spinner_forest = Spinner()  # For Random Forest Regression

        # Load and preprocess data
        print("Loading Data...")
        data = load_data()  # Load data using a predefined function
        data = data.fillna(data.mean())  # Fill missing values with mean

        print(data)  # Just for reference

        # Perform feature scaling on the data
        print("Performing Feature Scaling...")
        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        """
        START LINEAR REGRESSION
        """

        # Linear Regression
        print_section_header("Linear Regression")
        start_time_linear = time.time()  # Start timing

        # Data splitting
        print("Splitting Data into Training and Testing Sets...")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Linear regression settings
        learning_rate = 0.001
        num_iterations = 1000

        # Linear regression process
        print("Performing Linear Regression using Gradient Descent...")
        try:
            print("Trying to compute using CUDA GPU technology...")
            m, q = gradient_descent_gpu(train_data, learning_rate, num_iterations)
        except Exception as e:
            print(e)
            print("Unable to compute using parallelization.")
            print("Using CPU computation: Warning! Process will take longer...")
            m, q = gradient_descent(train_data, learning_rate, num_iterations)

        # Calculate MSE for linear regression
        print("Calculating Mean Squared Error for Linear Regression on Test Set...")
        mse_linear = calculate_linear_mse(test_data, m, q)
        stop_time_linear = time.time()
        execution_time_linear = stop_time_linear - start_time_linear

        """ 
        Preparing Linear ROC
        """

        # Calculates predictions for the test set
        linear_predictions = m * test_data['Numerical_Index_scaled'] + q

        # Establish a threshold for classification
        threshold = test_data['Close_scaled'].mean()
        # We use the average of the 'Close_scaled' values as the method for calculating the threshold.

        # Convert predictions to binary values
        binary_predictions = convert_to_binary(linear_predictions, threshold)

        # Calculates the true positive and false positive for different thresholds
        fpr, tpr, thresholds = roc_curve(test_data['Close_scaled'].apply(lambda x: 1 if x >= threshold else 0),
                                         binary_predictions)

        # Calculate Area Under the Curve (AUC)
        auc_value = auc(fpr, tpr)

        # Plot the single ROC curve  - *** delegated to the final section of the code ***
        """
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Linear Regression ROC (area = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for Linear Regression')
        plt.legend(loc="lower right")
        plt.show()
        """
        """
        END LINEAR REGRESSION
        """

        """
        START POLYNOMIAL REGRESSION
        """
        print_section_header("Polynomial Regression")
        start_time_poly = time.time()
        # Perform polynomial regression with a degree of 16 and get the coefficients
        print("Performing Polynomial Regression...")

        degree = 16  # degree of the polynomial
        coefficients = polynomial_regression(train_data, degree)

        # Calculate MSE for polynomial regression
        print("Calculating Mean Squared Error for Polynomial Regression on Test Set...")
        mse_polynomial = calculate_polynomial_mse(test_data, coefficients)
        stop_time_poly = time.time()
        execution_time_poly = stop_time_poly - start_time_poly

        """
        Preparing Polynomial ROC
        """

        # Calculates predictions for the test set
        polynomial_pred = polynomial_predictions(test_data, coefficients)

        # Establish a threshold for classification
        threshold = test_data['Close_scaled'].mean()  # the mean of the values 'Close_scaled'

        # Convert predictions to binary values
        binary_predictions = convert_to_binary(polynomial_pred, threshold)

        # Calculates the true positive and false positive for different thresholds
        fpr_pr, tpr_pr, thresholds = roc_curve(test_data['Close_scaled'].apply(lambda x: 1 if x >= threshold else 0),
                                         binary_predictions)

        # Calculate Area Under the Curve (AUC)
        auc_value_pr = auc(fpr_pr, tpr_pr)

        """
        # Plot the ROC curve *** delegated to the final section of the code ***
        plt.figure()
        plt.plot(fpr_pr, tpr_pr, color='darkorange', lw=2, label=f'Polynomial Regression ROC curve (area = {auc_value_pr:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for Polynomial Regression')
        plt.legend(loc="lower right")
        plt.show()
        """
        """
        END POLYNOMIAL REGRESSION
        """

        """
        START DECISION TREE REGRESSION
        """
        # Decision tree process
        print_section_header("Decision Tree Regression")
        start_time_tree = time.time()
        print("Performing Decision Tree Regression...")
        spinner_tree.start()

        # Decision Tree Regression
        # Separation of features and target
        X = data.drop('Close', axis=1).values  # Assuming 'Close' is target
        y = data['Close'].values

        # Division into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creation and training of the decision tree regression model
        tree_regressor = DecisionTreeRegressor(min_samples_split=2, max_depth=5)  # Editable parameters (2,5) is optimal
        tree_regressor.fit(X_train, y_train)

        # Predictions on test data
        y_pred_tree = tree_regressor.predict(X_test)

        # Calculation of mean square error (MSE)
        mse_tree = mean_squared_error(y_test, y_pred_tree)
        spinner_tree.stop()
        print("Tree completed.")
        stop_time_tree = time.time()
        execution_time_tree = stop_time_tree - start_time_tree

        """
        Preparing Decision Tree Regression ROC
        """

        # Decision tree predictions
        y_pred_tree = tree_regressor.predict(X_test)

        # Establish a threshold for classification
        threshold = np.mean(y_test)

        # Convert predictions to binary values
        binary_predictions = convert_to_binary(y_pred_tree, threshold)

        # Calculates the true positive and false positive for different thresholds
        fpr_dt, tpr_dt, thresholds = roc_curve(y_test >= threshold, binary_predictions)

        # Calculate Area Under the Curve (AUC)
        auc_value_dt = auc(fpr_dt, tpr_dt)

        # Plot the ROC curve *** delegated to the final section of the code ***
        """
        plt.figure()
        plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label=f'Tree Regression ROC curve (area = {auc_value_dt:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for Decision Tree Regression')
        plt.legend(loc="lower right")
        plt.show()
        """
        """
        END DECISION TREE REGRESSION
        """

        """
        START RANDOM FOREST REGRESSION
        """

        print_section_header("Random Forest Regression")
        # Record the start time for execution time calculation
        start_time_forest = time.time()
        print("Performing Random Forest Regression...")
        # Start a spinner (a simple text-based spinner animation) to indicate processing
        spinner_forest.start()

        # Creating and training the Random Forest model
        # n_estimators refers to the number of trees in the forest
        # min_samples_split specifies the minimum number of samples required to split an internal node
        # max_depth is the maximum depth of the trees
        random_forest = RandomForestRegressor(n_estimators=8, min_samples_split=2, max_depth=3)
        random_forest.fit(X_train, y_train)

        # Making predictions on the test dataset
        y_pred_forest = random_forest.predict(X_test)

        # Calculating Mean Squared Error (MSE) for model evaluation
        mse_forest = mean_squared_error(y_test, y_pred_forest)

        # Stopping the spinner animation
        spinner_forest.stop()
        print("Forest completed")

        # Calculating the execution time for the Random Forest Regression
        stop_time_forest = time.time()
        execution_time_forest = stop_time_forest - start_time_forest

        """
        Preparing Random Forest Regression ROC 
        """


        # Making predictions on the test dataset
        y_pred_forest = random_forest.predict(X_test)

        # Establish a threshold for classification
        # Here, the threshold is set as the mean of the actual test values
        threshold = np.mean(y_test)

        # Convert continuous predictions to binary (0/1) based on the threshold
        binary_predictions = convert_to_binary(y_pred_forest, threshold)

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for different threshold values
        fpr_rf, tpr_rf, thresholds = roc_curve(y_test >= threshold, binary_predictions)

        # Calculate Area Under the Curve (AUC)
        auc_value_rf = auc(fpr_rf, tpr_rf)

        """
        # Plotta la curva ROC
        plt.figure()
        plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'R_Forest Regression ROC curve (area = {auc_value_rf:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for Random Forest Regression')
        plt.legend(loc="lower right")
        plt.show()
        """

        """
        END RANDOM FOREST REGRESSION
        """

        """
        START ARTIFICIAL NEURAL NETWORK
        """
        start_time_ann = time.time()
        try:
            print("\n\nPerforming ANN Regression...")
            data = (data - data.mean()) / data.std()
            # Separation of characteristics and target
            X = data.drop('Close', axis=1).values  # Assuming that 'Close' is the target
            y = data['Close'].values.reshape(-1, 1)  # Reshape y to fit the dimensions expected by ANN

            # Division into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Defining the dimensions of the neural network
            input_size = X_train.shape[1]  # Number of features
            hidden_sizes = [20, 10, 5]  # Size of hidden layers
            output_size = 1  # Output size for regression

            # ANN initialisation and training
            ann_model = ImprovedNeuralNetwork([input_size] + hidden_sizes + [output_size], learning_rate=0.0015, epochs=5000)
            ann_model.train(X_train, y_train)

            # ANN evaluation
            y_pred_ann = ann_model.predict(X_test)
            mse_ann = mean_squared_error(y_test, y_pred_ann)

            # Optional display - ** Used for the analysis phase
            # ann_model.plot_loss()
            # ann_model.plot_predictions(X_test, y_test)

            """
            Plotting ROC
            """

            # ANN Predictions
            y_pred_ann = ann_model.predict(X_test)

            # Transforms predictions and actual values into a suitable format
            y_pred_ann_flat = y_pred_ann.flatten()
            y_test_flat = y_test.flatten()

            # Establish a threshold for classification
            threshold = np.mean(y_test_flat)  # average of real values in y_test_flat

            # Convert predictions to binary values
            binary_predictions = convert_to_binary(y_pred_ann_flat, threshold)

            # Calculates the true positive and false positive for different thresholds
            fpr_ann, tpr_ann, thresholds = roc_curve(y_test_flat >= threshold, binary_predictions)

            # Calculate Area Under the Curve (AUC)
            auc_value_ann = auc(fpr_ann, tpr_ann)

            """
            # Plotta la curva ROC
            plt.figure()
            plt.plot(fpr_ann, tpr_ann, color='darkorange', lw=2, label=f'ROC curve (area = {auc_value_ann:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for ANN Regression')
            plt.legend(loc="lower right")
            plt.show()
            """

            """
            Plt all ROC together
            """

            plt.figure(figsize=(10, 8))
            # Linear Regression ROC
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'Linear Regression (area = {auc_value:.2f})')

            # Polynomial Regression ROC
            plt.plot(fpr_pr, tpr_pr, color='red', lw=2, label=f'Polynomial Regression (area = {auc_value_pr:.2f})')

            # Decision Tree Regression ROC
            plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label=f'Decision Tree Regression (area = {auc_value_dt:.2f})')

            # Random Forest Regression ROC
            plt.plot(fpr_rf, tpr_rf, color='purple',linestyle="--", lw=2, label=f'Random Forest Regression (area = {auc_value_rf:.2f})')

            # ANN Regression ROC
            plt.plot(fpr_ann, tpr_ann, color='orange', lw=2, label=f'ANN Regression (area = {auc_value_ann:.2f})')

            # Plot details
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for All Regressions')
            plt.legend(loc="lower right")
            plt.show()

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

        """
        PRINT RESULT SECTION
        """

        """
        graph_spinner = Spinner()
        print("Plotting results...")

        # Start the spinner animation
        graph_spinner.start()

        # Calling the plot_decision_tree_regression function
        # To print this graph, we unfortunately have to re-train the model on the scaled and transformed
        # features to match the scaling values of the graph of the other features.
        # From this point on, we abandoned this method of representation as it slowed
        # the thread down a lot and did not bring any advantage.
        plot_combined_regression_with_decision_tree(train_data, test_data, coefficients, m, q, tree_regressor)

        # Stop the spinner animation as the plotting is complete
        graph_spinner.stop()
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

        input("\n\nPress Enter to Plot the graph.\n>")

        plot_graph(train_data, test_data, coefficients, m, q, tree_regressor)

    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print("Exception encountered:", e)


# Main Execution
if __name__ == '__main__':
    forecaster()
