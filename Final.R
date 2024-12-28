# Required libraries
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(pROC)
library(ggplot2)
library(dplyr)
library(ggcorrplot)

# Read and combine datasets
target_credit <- read.csv("target_credit_data.csv")
german_credit <- read.csv("german_credit_data.csv")
combined_credit <- bind_rows(target_credit, german_credit)

# Find missing value
total_missing <- sum(is.na(combined_credit))
print(paste("Total NA Value:", total_missing))

# Data preprocessing
combined_credit <- combined_credit %>%
  mutate_if(is.character, as.factor) %>%
  na.omit() %>%
  filter(!duplicated(.))

# Feature scaling for numeric columns
numeric_cols <- sapply(combined_credit, is.numeric)
numeric_data <- combined_credit[, numeric_cols]
range_model <- preProcess(numeric_data, method = c("range"))
numeric_data_scaled <- predict(range_model, numeric_data)
combined_credit[, numeric_cols] <- numeric_data_scaled

# Add this block to examine correlations
correlation_matrix <- cor(numeric_data_scaled)
ggcorrplot(correlation_matrix, method = "circle", 
           type = "lower", lab = TRUE, title = "Correlation Matrix")

# Count the occurrences of each Risk category
risk_counts <- combined_credit %>%
  group_by(Risk) %>%
  summarise(Count = n())

# Plot the Risk attribute distribution
ggplot(risk_counts, aes(x = Risk, y = Count, fill = Risk)) +
  geom_bar(stat = "identity", width = 0.5) +
  labs(title = "Comparison of Good and Bad Risk Counts",
       x = "Risk",
       y = "Count") +
  scale_fill_manual(values = c("Good" = "blue", "Bad" = "red")) +
  theme_minimal()


# Split data
set.seed(123)
train_index <- createDataPartition(combined_credit$Risk, p = 0.8, list = FALSE)
train_data <- combined_credit[train_index, ]
test_data <- combined_credit[-train_index, ]

# Define cross-validation settings
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train multiple models
models <- list(
  rf = train(Risk ~ ., data = train_data, method = "rf", 
             trControl = train_control, metric = "ROC"),
  svm = train(Risk ~ ., data = train_data, method = "svmRadial", 
             trControl = train_control, metric = "ROC"),
  xgb = train(Risk ~ ., data = train_data, method = "xgbTree", 
             trControl = train_control, metric = "ROC")
)

# Function to evaluate models
evaluate_model <- function(model, model_name, test_data) {
  # Predictions
  predictions <- predict(model, newdata = test_data)
  pred_probs <- predict(model, newdata = test_data, type = "prob")
  
  # Confusion Matrix
  conf_matrix <- confusionMatrix(predictions, test_data$Risk)
  
  # Calculate metrics
  precision <- conf_matrix$byClass["Pos Pred Value"]
  recall <- conf_matrix$byClass["Sensitivity"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # ROC and AUC
  roc_obj <- roc(as.numeric(test_data$Risk), pred_probs[,2])
  auc_val <- auc(roc_obj)
  
  # Print results
  cat("\n=== Model:", model_name, "===\n")
  print(conf_matrix)
  cat("\nAccuracy:", conf_matrix$overall["Accuracy"])
  cat("\nPrecision:", precision)
  cat("\nRecall:", recall)
  cat("\nF1-Score:", f1_score)
  cat("\nAUC:", auc_val, "\n")
  
  # Plot ROC curve
  plot(roc_obj, main = paste("ROC Curve -", model_name))
  
  return(list(
    accuracy = conf_matrix$overall["Accuracy"],
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    auc = auc_val
  ))
}

# Evaluate all models
results <- lapply(names(models), function(model_name) {
  evaluate_model(models[[model_name]], model_name, test_data)
})
names(results) <- names(models)

# Compare models
resamples <- resamples(models)
bwplot(resamples)
summary(resamples)

# Find best model based on AUC
best_model_name <- names(which.max(sapply(results, function(x) x$auc)))
cat("\nBest performing model based on AUC:", best_model_name)
