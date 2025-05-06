# 📦 Install and Load Required Packages
install.packages(c("dplyr", "ggplot2", "caret", "pROC", "corrplot", "MASS"))
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)
library(corrplot)
library(MASS)

# 📂 1. Load Data
data("GermanCredit")
credit_data <- GermanCredit

# Quick overview
glimpse(credit_data)

# Check missing values
sum(is.na(credit_data))

# 📊 2. Data Exploration

# 2.1 Class distribution
table(credit_data$Class)
prop.table(table(credit_data$Class))

# 2.2 Visualize Loan Amount Distribution by Class
ggplot(credit_data, aes(x = Class, y = Amount, fill = Class)) +
  geom_boxplot() +
  labs(title = "Loan Amount Distribution by Default Status", y = "Loan Amount") +
  theme_minimal()

# 2.3 Correlation Matrix for Numeric Variables
numeric_vars <- credit_data %>% select_if(is.numeric)
numeric_vars <- numeric_vars[, sapply(numeric_vars, function(x) sd(x) > 0)]  # Remove zero-variance columns
corrplot(cor(numeric_vars), method = "color", type = "upper", tl.cex = 0.7)

# 📈 3. Feature Selection
selected_vars <- credit_data %>%
  dplyr::select(Class, Age, Amount, Duration,
                Housing.Rent, Housing.Own, Housing.ForFree,
                Job.UnemployedUnskilled, Job.UnskilledResident, Job.SkilledEmployee, Job.Management.SelfEmp.HighlyQualified)

# 🔀 4. Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(selected_vars$Class, p = 0.8, list = FALSE)
trainData <- selected_vars[trainIndex, ]
testData <- selected_vars[-trainIndex, ]

# 🧠 5. Build Logistic Regression Model
logit_model <- glm(Class ~ ., data = trainData, family = binomial)
summary(logit_model)

# 🔮 6. Predict on Test Data
# Fresh prediction on test variables only
testData_small <- testData %>% 
  dplyr::select(Age, Amount, Duration,
                Housing.Rent, Housing.Own, Housing.ForFree,
                Job.UnemployedUnskilled, Job.UnskilledResident, Job.SkilledEmployee, Job.Management.SelfEmp.HighlyQualified)

pred_probs <- predict(logit_model, newdata = testData_small, type = "response")

# Default threshold = 0.5
pred_class <- ifelse(pred_probs > 0.5, "Bad", "Good")
pred_class <- factor(pred_class, levels = c("Good", "Bad"))

# 📏 7. Model Evaluation (Threshold = 0.5)

# 7.1 Confusion Matrix
confusionMatrix(pred_class, testData$Class)

# 7.2 ROC Curve and AUC
roc_curve <- roc(testData$Class, pred_probs, levels = c("Good", "Bad"))
plot(roc_curve, col = "blue", main = "ROC Curve")
auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 3)))

# --- 8. Feature Importance and Threshold Tuning ---

# --- 8.1 Feature Importance (Standardized Coefficients) ---

# Standardize continuous features in the training set
train_means <- trainData %>% summarise(across(c(Age, Amount, Duration), mean))
train_sds   <- trainData %>% summarise(across(c(Age, Amount, Duration), sd))

trainData_scaled <- trainData %>%
  mutate(
    Age = (Age - train_means$Age) / train_sds$Age,
    Amount = (Amount - train_means$Amount) / train_sds$Amount,
    Duration = (Duration - train_means$Duration) / train_sds$Duration
  )

# Refit logistic regression model on the standardized training data
logit_model_scaled <- glm(Class ~ ., data = trainData_scaled, family = binomial)

summary(logit_model_scaled)
# The coefficients in the summary of 'logit_model_scaled' are the standardized coefficients.
# Their magnitude gives an indication of the feature importance, relative to other standardized features.


# --- 8.2 Threshold Tuning (Find Best Threshold from Scaled Training Data ROC) ---

# 🔥 Build ROC on the scaled training data predictions
train_pred_scaled_probs <- predict(logit_model_scaled, newdata = trainData_scaled, type = "response")
roc_curve_scaled_train <- roc(trainData$Class, train_pred_scaled_probs, levels = c("Good", "Bad"))
best_threshold <- coords(roc_curve_scaled_train, "best", ret = "threshold")
print(paste("Best Threshold (from scaled training data):", round(best_threshold, 3)))


# --- 8.3 Prepare Test Data Correctly for Prediction with Scaled Model ---

# Select the predictor variables from the test set and scale them
testData_for_pred_scaled <- testData %>%
  dplyr::select(Age, Amount, Duration,
                Housing.Rent, Housing.Own, Housing.ForFree,
                Job.UnemployedUnskilled, Job.UnskilledResident, Job.SkilledEmployee, Job.Management.SelfEmp.HighlyQualified) %>%
  mutate(
    Age = (Age - train_means$Age) / train_sds$Age,
    Amount = (Amount - train_means$Amount) / train_sds$Amount,
    Duration = (Duration - train_means$Duration) / train_sds$Duration
  )


# --- 8.4 Make Predictions using the Scaled Model on Scaled Test Data ---

# Check dimensions of the data being used for prediction (important for debugging)
cat("Dimensions of testData_for_pred_scaled:", dim(testData_for_pred_scaled), "\n")

# Predict probabilities on the scaled test data
pred_probs_scaled_test <- predict(logit_model_scaled, newdata = testData_for_pred_scaled, type = "response")

# Check prediction length (should match the number of rows in testData)
cat("Length of predictions (scaled test data):", length(pred_probs_scaled_test), "\n")


# --- 8.5 Evaluate Model Performance with Tuned Threshold ---

# Explicitly ensure best_threshold is a single numeric value
best_threshold_numeric <- as.numeric(best_threshold)
cat("Best Threshold (numeric):", best_threshold_numeric, "\n")

# Double-check the length of prediction probabilities
cat("Length of pred_probs_scaled_test:", length(pred_probs_scaled_test), "\n")

# Apply the best threshold and create the predicted class labels
pred_class_best_scaled <- factor(ifelse(pred_probs_scaled_test > best_threshold_numeric, "Bad", "Good"), levels = c("Good", "Bad"))

# True labels from testData
true_labels <- factor(testData$Class, levels = c("Good", "Bad"))

# Check lengths (crucial for confusionMatrix)
cat("Length of predictions (tuned threshold):", length(pred_class_best_scaled), "\n")
cat("Length of true labels:", length(true_labels), "\n")

# Confusion Matrix with the tuned threshold
confusionMatrix(pred_class_best_scaled, true_labels)

# ROC Curve and AUC on the test data with the scaled model and tuned threshold
roc_curve_test_scaled <- roc(testData$Class, pred_probs_scaled_test, levels = c("Good", "Bad"))
plot(roc_curve_test_scaled, col = "darkgreen", main = "ROC Curve (Scaled Model on Test Data with Tuned Threshold)")
auc_value_scaled_test <- auc(roc_curve_test_scaled)
print(paste("AUC (Scaled Model on Test Data with Tuned Threshold):", round(auc_value_scaled_test, 3)))