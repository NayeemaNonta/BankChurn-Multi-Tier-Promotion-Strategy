# Logistic Regression
library(glmnet)

x_train <- read.csv("/Users/yalichen/Documents/MSCI433/Project/x_train.csv")
x_test <- read.csv("/Users/yalichen/Documents/MSCI433/Project/x_test.csv")
y_train <- read.csv("/Users/yalichen/Documents/MSCI433/Project/y_train.csv")
y_test <- read.csv("/Users/yalichen/Documents/MSCI433/Project/y_test.csv")

# Convert y_train to a factor
y_train <- as.factor(y_train$churn)

# Combine X_train and y_train into a single data frame
train_data <- cbind(y_train, x_train)

# Fit logistic regression model
logit_model <- glm(y_train ~ ., data = train_data, family = "binomial")
summary(logit_model)

# Predict on the test set
predictions <- predict(logit_model, newdata = x_test, type = "response")

# Convert predicted probabilities to binary predictions (0 or 1)
binary_predictions <- ifelse(predictions > 0.2, 1, 0)

# Sensitivity (TP) Specificity (TN)
y_test_vector <- y_test$churn

# Check the class of y_test_vector
class(y_test_vector)

# Now, create the confusion matrix
conf_matrix <- table(binary_predictions, y_test_vector)
conf_matrix
TN <- conf_matrix[1, 1]
FP <- conf_matrix[1, 2]
FN <- conf_matrix[2, 1]
TP <- conf_matrix[2, 2]
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
accuracy <- (TP + TN) / sum(conf_matrix)

print("Confusion Matrix:")
print(conf_matrix)
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))
print(paste("Precision:", precision))
print(paste("Accuracy:", accuracy))


# Define threshold values and corresponding metric values
threshold <- c(0.2, 0.5, 0.7)
sensitivity <- c(35.80, 77.97, 81.82)/100
accuracy <- c(70.10, 81.24, 80.88)/100
precision <- c(67.1, 5.35, 2.61)/100

# Create a matrix containing metric values
metric_values <- rbind(accuracy, precision,sensitivity)

# Define metric names
metric_names <- c("Accuracy", "Precision","Sensitivity")

# Plot bar chart
barplot(metric_values, beside = TRUE, col = c("blue", "orange","green" ),
        main = "Logistic Regression Metrics for Different Thresholds", xlab = "Threshold", ylab = "Metric Value",
        legend.text = metric_names, args.legend = list(x = "topleft",cex = 0.4),
        names.arg = threshold)




# Calculate true negatives (TN) and false positives (FP) for baseline model
TN_baseline <- sum(y_test == 0)
FP_baseline <- sum(y_test == 1)
accuracy_baseline <- TN_baseline / (TN_baseline + FP_baseline)

print(paste("Accuracy of Baseline Model:", accuracy_baseline))

# Plot ROC
library(ROCR)
test_probabilities <- predict(logit_model, newdata = x_test, type = "response")
roc_curve <- prediction(test_probabilities,y_test)
ROCRperf = performance(roc_curve, "tpr","fpr")
plot(ROCRperf)
plot(ROCRperf, colorize = TRUE)
plot(ROCRperf, colorize = TRUE, print.cutoffs.at = seq(0,1,0.1))
plot(ROCRperf, colorize = TRUE, print.cutoffs.at = seq(0,1,0.2), text.adj=c(-0.2,2.0))

# Compute AUC
library(pROC)
str(roc_curve)
# Extract TPR and FPR
tpr <- ROCRperf@y.values[[1]]
fpr <- ROCRperf@x.values[[1]]

# Compute AUC
roc_auc <- sum(diff(fpr) * tpr[-length(tpr)])
cat("AUC:", roc_auc, "\n")


# Decision Tree
library(rpart)
library(rpart.plot)
train_data <- cbind(y_train, x_train)

# Fit decision tree model
tree_model <- rpart(y_train ~ ., data = train_data, method = "class")

# Make predictions on the test set
predictions <- predict(tree_model, newdata = x_test, type = "class")
predictions_prob <- predict(tree_model, newdata = x_test, type = "prob")

# Ensure y_test is treated as a vector
y_test <- as.vector(y_test)
y_test_vector <- as.vector(y_test$churn)

library(pROC)
roc_result <- roc(y_test_vector, predictions_prob[,2])
auc_value <- auc(roc_result)

# Evaluate performance
accuracy <- mean(predictions == y_test_vector)
print(paste("Accuracy of Decision Tree Model:", accuracy))
print(paste("AUC of the Decision Tree model:", auc_value))

rpart.plot(tree_model, yesno = 2, type = 2, extra = 101, under = TRUE, cex = 0.8)
