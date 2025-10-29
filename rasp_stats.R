# clear environment
rm(list=ls())

# ---- Target - Background----

# Load a CSV file into R
data_diff <- read.csv('/mnt/storage/RASP/results/merged_stat_diff_plvs.csv')

# View the first few rows
head(data_diff)

# add the significant cluster
sigclu_diff <- c('FC5', 'C5', 'C3', 'C1', 'CP5', 'CP3', 'P5', 'PO7', 'POz', 'PO4','PO8', 'O2')   # based on python output
data_diff$mean_sigclu <- rowMeans(data_diff[, sigclu_diff], na.rm = TRUE)
head(data_diff$mean_sigclu)

# Run one-way ANOVA
anova_diff <- aov(mean_sigclu ~ Condition, data = data_diff)

# View the summary of the ANOVA
summary(anova_diff)

library(ggplot2)
library(dplyr)

ggplot(data_diff, aes(x = Condition, y = mean_sigclu)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "Difference in PLVs by Condition",
       x = "Condition", y = "Target - Background PLV") +
  theme_minimal()

posthoc_diff <- TukeyHSD(anova_diff)

# View the results
print(posthoc_diff)

# Run multiple regression with interaction
model_diff <- lm(behaviorscore ~ mean_sigclu * Condition, data = data_diff)

# View the summary
summary(model_diff)

# visualize the model
ggplot(data_diff, aes(x = mean_sigclu, y = behaviorscore, color = Condition)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) + labs(title = "Behavior Score vs Difference in PLV by Condition",
       x = "Difference in PLV", y = "Behavior Score")

#---- Behavior results----
# Run one-way ANOVA
anova_behavior <- aov(behaviorscore ~ Condition, data = data_diff)
# View the summary of the ANOVA
summary(anova_behavior)

posthoc_behavior <- TukeyHSD(anova_behavior)
# View the results
print(posthoc_behavior)

# Behavior visualization
ggplot(data_diff, aes(x = Condition, y = behaviorscore)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "Behavhior Score by Condition",
       x = "Condition", y = "Percent Correct") +
  theme_minimal()


#---- Target Stream----
# Load a CSV file into R
data_ta <- read.csv('/mnt/storage/RASP/results/merged_stat_ta_plvs.csv')

# View the first few rows
head(data_ta)

# add the significant cluster
chall_ta <-c('Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FC5','FC3','FC1','FCz','FC2','FC4','FC6','T7', 'C5', 'C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','M1','M2','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2')

data_ta$mean_chall <- rowMeans(data_ta[, chall_ta], na.rm = TRUE)

# Run one-way ANOVA
anova_ta <- aov(mean_chall ~ Condition, data = data_ta)

# View the summary of the ANOVA
summary(anova_ta)

ggplot(data_ta, aes(x = Condition, y = mean_chall)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "PLV Target by Condition",
       x = "Condition", y = "PLV Target") +
  theme_minimal()

posthoc_ta <- TukeyHSD(anova_ta)

# View the results
print(posthoc_ta)

# Run multiple regression with interaction
model <- lm(log(behaviorscore) ~ mean_chall * Condition, data = data_ta)

# View the summary
summary(model)

ggplot(data_ta, aes(x = mean_chall, y = behaviorscore, color = Condition)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) + labs(title = "Behavior Score vs PLV Target by Condition",
                                                x = "PLV Target", y = "Behavior Score")


#---- Background Stream----
# Load a CSV file into R
data_bg <- read.csv("/mnt/storage/RASP/results/merged_stat_bg_plvs.csv")

# View the first few rows
head(data_bg)

# add the significant cluster
sigclu_bg <- c('Fp1', 'Fp2', 'AF3', 'F1', 'Fz', 'F2', 'F8', 'FC5', 'FC3', 'FC1',
  'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
  'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP6', 'TP8', 'P7', 'P5', 'P3',
  'P1', 'Pz', 'P6', 'PO7', 'PO3', 'POz', 'PO8', 'O1')

chall_bg <-c('Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FC5','FC3','FC1','FCz','FC2','FC4','FC6','T7', 'C5', 'C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','M1','M2','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2')

data_bg$mean_sigclu <- rowMeans(data_bg[, sigclu_bg], na.rm = TRUE)

data_bg$mean_chall <- rowMeans(data_bg[, chall_bg], na.rm = TRUE)

# Run one-way ANOVA
anova_bg <- aov(mean_chall ~ Condition, data = data_bg)

# View the summary of the ANOVA
summary(anova_bg)

ggplot(data_bg, aes(x = Condition, y = mean_)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "PLV by Condition",
       x = "Condition", y = "Background PLV") +
  theme_minimal()


posthoc_bg <- TukeyHSD(anova_bg)

# View the results
print(posthoc_bg)

# Run multiple regression with interaction
model <- lm(behaviorscore ~ mean_chall * Condition, data = data_bg)

# View the summary
summary(model)

ggplot(data_bg, aes(x = mean_chall, y = behaviorscore, color = Condition)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) + labs(title = "Behavior Score vs PLV Background by Condition",
                                                x = "PLV Background", y = "Behavior Score")



# ---- Load EEG data only 17 participants -----
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)

# Load a CSV file into R
eeg_ta <- read.csv("/mnt/storage/RASP/results/eeg_stat_ta_gamma_cohs.csv")
eeg_bg <- read.csv("/mnt/storage/RASP/results/eeg_stat_bg_gamma_cohs.csv")
eeg_diff <- read.csv("/mnt/storage/RASP/results/eeg_stat_diff_gamma_cohs.csv")


chall<-c('Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FC5','FC3','FC1','FCz','FC2','FC4','FC6','T7', 'C5', 'C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','M1','M2','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2')

eeg_ta$mean_chall <- rowMeans(eeg_ta[, chall], na.rm = TRUE)
eeg_bg$mean_chall <- rowMeans(eeg_bg[, chall], na.rm = TRUE)
eeg_diff$mean_chall <- rowMeans(eeg_diff[, chall], na.rm = TRUE)

data_all <- data.frame(
  mean_chall_ta = eeg_ta$mean_chall,
  mean_chall_bg = eeg_bg$mean_chall)

data_all$Condition <- eeg_bg$Condition
data_all$participant <- eeg_bg$participant

# for visualizing ta and bg together
data_filtered <- data_all %>%
  filter(Condition %in% c("TA", "CT", "BA")) %>%
  select(participant, Condition, mean_chall_ta, mean_chall_bg)

# Reshape to long format
data_long <- data_filtered %>%
  pivot_longer(cols = c(mean_chall_ta, mean_chall_bg),
               names_to = "Stream",
               values_to = "MeanChall")

ggplot(data_long, aes(x = Condition, y = MeanChall, fill = Stream)) +
  geom_violin(position = position_dodge(width = 0.8), alpha = 0.5) +
  geom_jitter(aes(color = Stream),
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 1.5, alpha = 0.7) +
  labs(title = "Mean PLVs by Condition and Stream",
       x = "Condition", y = "Mean Across All Channels") +
  theme_minimal()

# Fit the model
model <- lmer(MeanChall ~ Condition * Stream + (1 | participant), data = data_long)

# Summary of the model
summary(model)

## for diff visualization
data_filtered_diff <- eeg_diff %>%
  filter(Condition %in% c("TA", "CT", "BA")) %>%
  select(participant, Condition, mean_chall)


ggplot(data_filtered_diff, aes(x = Condition, y = mean_chall)) +
  geom_violin(trim = FALSE, fill = "lightblue", color = "black") +
  geom_jitter(width = 0.2, size = 2, alpha = 0.6, color = "darkblue") +
  labs(title = "Distribution of Mean PLVs Across Channels by Conditions",
       x = "Condition",
       y = "Mean PLVs") +
  theme_minimal()



model_diff <- lmer(mean_chall ~ Condition + (1 | participant), data = data_filtered_diff)

# Summary of the model
summary(model_diff)

data_long_diff <- data_filtered_diff %>%
  pivot_longer(cols = c(mean_chall),
               names_to = "Difference",
               values_to = "MeanChall")


## take channel in as a fixed factor
# Reshape to long format
data_long <- eeg_diff %>%
  pivot_longer(cols = 4:61,
               names_to = "Channel",
               values_to = "Value")

ggplot(data_long, aes(x = Channel, y = Value, fill = Condition)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5) +
  geom_jitter(aes(color = Condition),
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
               size = 1.5, alpha = 0.7) +
   labs(title = "Mean PLVs by Condition and Stream",
        x = "Condition", y = "Mean Across All Channels") +
   theme_minimal()

# Fit the model
model <- lmer(Value ~ Condition * Channel + (1 | participant), data = data_long)

# Summary of the model
summary(model)

# Extract summary table
model_summary <- summary(model)

# Get fixed effects table
fixed_effects <- as.data.frame(model_summary$coefficients)

# Filter for significant effects
significant_effects <- fixed_effects[fixed_effects$`Pr(>|t|)` < 0.05, ]

# View results
print(significant_effects)

emm <- emmeans(model, ~ Condition | Channel)

# Pairwise comparisons across Condition within each Channel
posthoc <- pairs(emm, adjust = "tukey")

# View significant results only
summary(posthoc) %>% filter(p.value < 0.05)

## now analyze EEG along with behavior

# ---- Target - Background----

# Load a CSV file into R
merged_diff <- read.csv('/mnt/storage/RASP/results/merged_stat_diff_gamma_plvs.csv')

# View the first few rows
head(merged_diff)

# add the significant cluster
sigclu_diff <- c('CP5')   # based on python output
merged_diff$mean_sigclu <- merged_diff[, sigclu_diff]
head(merged_diff$mean_sigclu)

# Run one-way ANOVA
anova_diff <- aov(mean_sigclu ~ Condition, data = merged_diff)

# View the summary of the ANOVA
summary(anova_diff)

library(ggplot2)
library(dplyr)

ggplot(merged_diff, aes(x = Condition, y = mean_sigclu)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "Difference in PLVs by Condition",
       x = "Condition", y = "Target - Background PLV") +
  theme_minimal()

posthoc_diff <- TukeyHSD(anova_diff)

# View the results
print(posthoc_diff)

# Run multiple regression with interaction
model_diff <- lm(behaviorscore ~ mean_sigclu * Condition, data = merged_diff)

# View the summary
summary(model_diff)

# visualize the model
ggplot(merged_diff, aes(x = mean_sigclu, y = behaviorscore, color = Condition)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) + labs(title = "Behavior Score vs Difference in PLV by Condition",
                                                x = "Difference in PLV", y = "Behavior Score")

#---- Behavior results----
# Run one-way ANOVA
anova_behavior <- aov(behaviorscore ~ Condition, data = data_diff)
# View the summary of the ANOVA
summary(anova_behavior)

posthoc_behavior <- TukeyHSD(anova_behavior)
# View the results
print(posthoc_behavior)

# Behavior visualization
ggplot(data_diff, aes(x = Condition, y = behaviorscore)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "Behavhior Score by Condition",
       x = "Condition", y = "Percent Correct") +
  theme_minimal()
