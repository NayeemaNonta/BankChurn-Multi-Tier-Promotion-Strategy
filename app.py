from pulp import LpVariable, LpProblem, LpMinimize, lpSum, value
import streamlit as st
import pandas as pd

# Define the optimization model function
def optimize_model(churn_probabilities, B, C, w, lower_threshold, upper_threshold):
    # Decision Variables
    customers = range(len(churn_probabilities))
    x_100 = LpVariable.dicts("x_100", customers, cat="Binary")  # Customer receives promotion $100
    x_200 = LpVariable.dicts("x_200", customers, cat="Binary")  # Customer receives promotion $200

    # Objective Function
    model = LpProblem("CustomerPromotionOptimization", LpMinimize)
    model += w * lpSum(churn_probabilities[i] * (1 - (x_100[i] + x_200[i])) for i in customers) \
             + (1 - w) * (lpSum(x_100[i] * C[100] + x_200[i] * C[200] for i in customers) / B)

    # Constraints
    model += lpSum(x_100[i] * C[100] + x_200[i] * C[200] for i in customers) <= B, "TotalPromotionBudget"
    for i in customers:
        if lower_threshold <= churn_probabilities[i] <= upper_threshold:
            model += x_100[i] == 1, f"Promotion100_{i}"
        else:
            model += x_200[i] == 1, f"Promotion200_{i}"

    # Solve the optimization problem
    model.solve()

    # Output results
    objective_value = value(model.objective)
    decision_variables = {
        "x_100": [value(x_100[i]) for i in customers],
        "x_200": [value(x_200[i]) for i in customers]
    }

    return objective_value, decision_variables

# Define Streamlit app
def main():
    st.title("Customer Promotion Optimization")

    # Load data
    df = pd.read_csv('predictions.csv')
    churn_probabilities = df['Churn_Probability'].values

    # Input parameters
    B = st.number_input("Total Promotion Budget", min_value=0)
    C = {100: 100, 200: 200}  # Cost of promotions (could be made customizable)
    w = st.slider("Weight assigned to the budget constraint", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    lower_threshold = st.number_input("Lower churn rate threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    upper_threshold = st.number_input("Upper churn rate threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

    # Optimize the model when the user clicks the button
    if st.button("Optimize"):
        objective_value, decision_variables = optimize_model(churn_probabilities, B, C, w, lower_threshold, upper_threshold)

        # Output results
        st.write("Objective Value:", objective_value)
        st.write("Values of decision variables x_100 and x_200:")
        decision_variable_results = {}
        for i in range(len(churn_probabilities)):
            decision_variable_results[f"Customer {i} receives promotion $100"] = decision_variables["x_100"][i]
            decision_variable_results[f"Customer {i} receives promotion $200"] = decision_variables["x_200"][i]
        st.write(decision_variable_results)

if __name__ == "__main__":
    main()
