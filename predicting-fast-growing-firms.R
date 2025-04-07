rm(list=ls())
# Fast_growth_prediction.R - Predicting Fast-Growing Firms (Assignment 3)

# Load required libraries
library(dplyr)
library(caret)
library(ggplot2)
library(knitr)
library(tidyverse)

# Load the full panel data (2010–2015)
panel_df <- read_csv("D:/CEU_MA_EDP/Winter 2025/Prediction/Assignment 3/cs_bisnode_panel.csv")


#####                            Task 1    _________________________________________
# How many years are covered?
table(panel_df$year)
# How many firm-year observations per firm?
panel_df %>% 
  count(comp_id) %>% 
  summarise(min = min(n), max = max(n), median = median(n))
# View a sample
panel_df %>% filter(comp_id %in% sample(unique(comp_id), 5))
# Filter for needed years and ensure sales > 0
sales_df <- panel_df %>%
  filter(year %in% c(2012, 2014)) %>%
  filter(!is.na(sales) & sales > 0)
# Log sales
sales_df <- sales_df %>%
  mutate(sales_log = log(sales)) %>%
  select(comp_id, year, sales_log)
# Reshape: wide format to compute growth
sales_wide <- sales_df %>%
  pivot_wider(names_from = year, values_from = sales_log, names_prefix = "sales_log_")
# Drop rows with missing sales in either year
sales_wide <- sales_wide %>%
  filter(!is.na(sales_log_2012) & !is.na(sales_log_2014)) %>%
  mutate(growth = sales_log_2014 - sales_log_2012)
# Define top 20% as fast-growing
threshold <- quantile(sales_wide$growth, 0.80, na.rm = TRUE)
sales_wide <- sales_wide %>%
  mutate(fast_growth = if_else(growth >= threshold, 1, 0))
# Preview
head(sales_wide)

# Filter 2012 data
model_df <- panel_df %>%
  filter(year == 2012)
# Join with fast_growth label
model_df <- model_df %>%
  left_join(sales_wide %>% select(comp_id, fast_growth), by = "comp_id")
# Keep only firms with a defined target
model_df <- model_df %>%
  filter(!is.na(fast_growth))
# Optional — select useful features (you can modify as needed)
model_df <- model_df %>%
  select(comp_id, fast_growth, sales, curr_assets, fixed_assets, liq_assets,
         profit_loss_year, personnel_exp, ceo_count, foreign, female,
         birth_year, inoffice_days, region_m, ind2, ind)
# Final check
glimpse(model_df)

### Logit Model ___________________
# Clean the data: remove rows with missing values
logit_data <- model_df %>%
  drop_na() %>%
  mutate(fast_growth = factor(fast_growth, levels = c(0, 1), labels = c("No", "Yes")))
# Train logit model again
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
logit_model <- train(
  fast_growth ~ . -comp_id,
  data = logit_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)
# Set outcome as factor
logit_data$fast_growth <- factor(logit_data$fast_growth)
# Print results
print(logit_model)

###  CART ---------------------------
# Set up cross-validation again
set.seed(123)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
# Train CART model
cart_model <- train(
  fast_growth ~ . -comp_id,
  data = logit_data,  # same cleaned dataset
  method = "rpart",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 10  # try multiple tree sizes
)
# View results
print(cart_model)
plot(cart_model)

###   Random Forest ______________________
# Ensure data is clean and fast_growth is a factor
rf_data <- logit_data  # already cleaned
# Set seed for reproducibility
set.seed(123)
# Train Random Forest model
rf_model <- train(
  fast_growth ~ . -comp_id,
  data = rf_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5
)
# View results
print(rf_model)
plot(rf_model)

###    Loss Function _________________

# Logit Threshold Tuning
# Extract the processed training data used internally
logit_data_used <- logit_model$trainingData
# Predict probabilities using this internal data
logit_probs <- predict(logit_model, newdata = logit_data_used, type = "prob")[, "Yes"]
# Extract actual labels
logit_actual <- logit_data_used$.outcome  # Already a factor with "Yes"/"No"
# Set thresholds
thresholds <- seq(0.01, 0.99, by = 0.01)
loss_data <- data.frame(threshold = thresholds, expected_loss = NA)
# Loop through thresholds to compute expected loss
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(logit_probs > threshold, "Yes", "No")
  cm <- table(Predicted = preds, Actual = logit_actual)
  
  TP <- cm["Yes", "Yes"]
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  
  FP <- ifelse(is.na(FP), 0, FP)
  FN <- ifelse(is.na(FN), 0, FN)
  
  loss_data$expected_loss[i] <- (FP * 1) + (FN * 10)
}
# Best threshold
best_logit_threshold <- loss_data %>% filter(expected_loss == min(expected_loss))
# Plot expected loss curve
library(ggplot2)
ggplot(loss_data, aes(x = threshold, y = expected_loss)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = best_logit_threshold$threshold, linetype = "dashed", color = "red") +
  labs(title = "Expected Loss vs Threshold (Logit Model)", y = "Expected Loss ($)", x = "Threshold") +
  theme_minimal()
# Show best threshold and loss
best_logit_threshold

# CART Threshold Tuning
# Predict probabilities using training data stored in CART model
cart_data_used <- cart_model$trainingData
cart_probs <- predict(cart_model, newdata = cart_data_used, type = "prob")[, "Yes"]
cart_actual <- cart_data_used$.outcome
# Set thresholds
thresholds <- seq(0.01, 0.99, by = 0.01)
cart_loss_data <- data.frame(threshold = thresholds, expected_loss = NA)
# Loop for expected loss
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(cart_probs > threshold, "Yes", "No")
  # Force full 2x2 confusion matrix structure
  cm <- table(factor(preds, levels = c("No", "Yes")),
              factor(cart_actual, levels = c("No", "Yes")))
  
  TP <- cm["Yes", "Yes"]
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  
  FP <- ifelse(is.na(FP), 0, FP)
  FN <- ifelse(is.na(FN), 0, FN)
  
  cart_loss_data$expected_loss[i] <- (FP * 1) + (FN * 10)
}
# Find best threshold
best_cart_threshold <- cart_loss_data %>% filter(expected_loss == min(expected_loss))
# Plot
ggplot(cart_loss_data, aes(x = threshold, y = expected_loss)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_vline(xintercept = best_cart_threshold$threshold, linetype = "dashed", color = "red") +
  labs(title = "Expected Loss vs Threshold (CART Model)", y = "Expected Loss ($)", x = "Threshold") +
  theme_minimal()
# Print best result
best_cart_threshold

# Random Forest Threshold Tuning
# Extract training data used inside RF model
rf_data_used <- rf_model$trainingData
rf_probs <- predict(rf_model, newdata = rf_data_used, type = "prob")[, "Yes"]
rf_actual <- rf_data_used$.outcome
# Initialize loss storage
thresholds <- seq(0.01, 0.99, by = 0.01)
rf_loss_data <- data.frame(threshold = thresholds, expected_loss = NA)
# Loop to calculate expected loss
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(rf_probs > threshold, "Yes", "No")
  
  cm <- table(factor(preds, levels = c("No", "Yes")),
              factor(rf_actual, levels = c("No", "Yes")))
  
  TP <- cm["Yes", "Yes"]
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  
  FP <- ifelse(is.na(FP), 0, FP)
  FN <- ifelse(is.na(FN), 0, FN)
  
  rf_loss_data$expected_loss[i] <- (FP * 1) + (FN * 10)
}
# Best threshold
best_rf_threshold <- rf_loss_data %>% filter(expected_loss == min(expected_loss))
# Plot
ggplot(rf_loss_data, aes(x = threshold, y = expected_loss)) +
  geom_line(color = "orange", linewidth = 1) +
  geom_vline(xintercept = best_rf_threshold$threshold, linetype = "dashed", color = "red") +
  labs(title = "Expected Loss vs Threshold (Random Forest)", y = "Expected Loss ($)", x = "Threshold") +
  theme_minimal()
# Show best threshold and loss
best_rf_threshold

### Generate Confusion Tables at Best Thresholds  __________________
# LOGIT CONFUSION TABLE at 0.09
logit_preds <- ifelse(logit_probs > 0.09, "Yes", "No")
logit_cm <- table(Predicted = factor(logit_preds, levels = c("No", "Yes")),
                  Actual = factor(logit_actual, levels = c("No", "Yes")))
cat("\n--- Logit Confusion Table (Threshold = 0.09) ---\n")
print(logit_cm)
# CART CONFUSION TABLE at 0.05 (midpoint of flat minimum)
cart_preds <- ifelse(cart_probs > 0.05, "Yes", "No")
cart_cm <- table(Predicted = factor(cart_preds, levels = c("No", "Yes")),
                 Actual = factor(cart_actual, levels = c("No", "Yes")))
cat("\n--- CART Confusion Table (Threshold = 0.05) ---\n")
print(cart_cm)
# RANDOM FOREST CONFUSION TABLE at 0.40
rf_preds <- ifelse(rf_probs > 0.40, "Yes", "No")
rf_cm <- table(Predicted = factor(rf_preds, levels = c("No", "Yes")),
               Actual = factor(rf_actual, levels = c("No", "Yes")))
cat("\n--- Random Forest Confusion Table (Threshold = 0.40) ---\n")
print(rf_cm)

####   Model Comparison Table  ______________________
# Create model performance table
model_results <- data.frame(
  Model = c("Logit", "CART", "Random Forest"),
  AUC_ROC = c(0.687, 0.674, 0.757),
  Sensitivity = c(0.998, 1.000, 1.000),
  Specificity = c(0.006, 0.000, 0.999),
  Best_Threshold = c(0.09, 0.05, 0.40),
  Expected_Loss = c(10540, 11061, 7)
)
# Print as a markdown table
kable(model_results, caption = "Model Performance Comparison (Overall)")


####                       Task 2 ______________________________________________
names(model_df)
table(model_df$ind2)

# Define Industry Groups in Code
# Manufacturing: ind2 from 10 to 33
manufacturing_df <- model_df %>%
  filter(ind2 >= 10 & ind2 <= 33) %>%
  drop_na() %>%
  mutate(fast_growth = factor(fast_growth, levels = c(0, 1), labels = c("No", "Yes")))

# Services: ind2 in 55, 56, or 95
services_df <- model_df %>%
  filter(ind2 %in% c(55, 56, 95)) %>%
  drop_na() %>%
  mutate(fast_growth = factor(fast_growth, levels = c(0, 1), labels = c("No", "Yes")))

# Train Random Forest (Manufacturing)
# Set up 5-fold CV
set.seed(123)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
# Train RF on Manufacturing firms
rf_manufacturing <- train(
  fast_growth ~ . -comp_id -ind2 -ind -region_m,
  data = manufacturing_df,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5
)
# Print result
print(rf_manufacturing)

# Train Random Forest (Services)
# Train RF on Services firms
rf_services <- train(
  fast_growth ~ . -comp_id -ind2 -ind -region_m,
  data = services_df,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5
)
# Print result
print(rf_services)

### Manufacturing — Threshold Tuning
# Predict probabilities using internal training data
manu_data_used <- rf_manufacturing$trainingData
manu_probs <- predict(rf_manufacturing, newdata = manu_data_used, type = "prob")[, "Yes"]
manu_actual <- manu_data_used$.outcome
# Initialize storage
thresholds <- seq(0.01, 0.99, by = 0.01)
manu_loss_data <- data.frame(threshold = thresholds, expected_loss = NA)
# Calculate expected loss at each threshold
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(manu_probs > threshold, "Yes", "No")
  
  cm <- table(factor(preds, levels = c("No", "Yes")),
              factor(manu_actual, levels = c("No", "Yes")))
  
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  
  FP <- ifelse(is.na(FP), 0, FP)
  FN <- ifelse(is.na(FN), 0, FN)
  
  manu_loss_data$expected_loss[i] <- (FP * 1) + (FN * 10)
}
# Best threshold
best_manu_threshold <- manu_loss_data %>% filter(expected_loss == min(expected_loss))
# Plot
library(ggplot2)
ggplot(manu_loss_data, aes(x = threshold, y = expected_loss)) +
  geom_line(color = "darkorange", linewidth = 1) +
  geom_vline(xintercept = best_manu_threshold$threshold, linetype = "dashed", color = "red") +
  labs(title = "Expected Loss vs Threshold (Manufacturing)", y = "Expected Loss ($)", x = "Threshold") +
  theme_minimal()
# Show best
best_manu_threshold

# Services — Threshold Tuning
# Predict probabilities using internal training data
serv_data_used <- rf_services$trainingData
serv_probs <- predict(rf_services, newdata = serv_data_used, type = "prob")[, "Yes"]
serv_actual <- serv_data_used$.outcome
# Initialize storage
serv_loss_data <- data.frame(threshold = thresholds, expected_loss = NA)
# Calculate expected loss
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  preds <- ifelse(serv_probs > threshold, "Yes", "No")
  
  cm <- table(factor(preds, levels = c("No", "Yes")),
              factor(serv_actual, levels = c("No", "Yes")))
  
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  
  FP <- ifelse(is.na(FP), 0, FP)
  FN <- ifelse(is.na(FN), 0, FN)
  
  serv_loss_data$expected_loss[i] <- (FP * 1) + (FN * 10)
}
# Best threshold
best_serv_threshold <- serv_loss_data %>% filter(expected_loss == min(expected_loss))
# Plot
ggplot(serv_loss_data, aes(x = threshold, y = expected_loss)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_vline(xintercept = best_serv_threshold$threshold, linetype = "dashed", color = "red") +
  labs(title = "Expected Loss vs Threshold (Services)", y = "Expected Loss ($)", x = "Threshold") +
  theme_minimal()
# Show best
best_serv_threshold

###      Industry Group Comparison Table
# Create the industry comparison table
industry_comparison <- data.frame(
  Sector = c("Manufacturing", "Services"),
  AUC_ROC = c(0.692, 0.784),
  Best_Threshold_Range = c("0.34–0.56", "0.39–0.55"),
  Expected_Loss = c(0, 0),
  Model_Accuracy = c("Perfect", "Perfect")
)
# Print table
kable(industry_comparison, caption = "Performance of Random Forest by Industry Group")



