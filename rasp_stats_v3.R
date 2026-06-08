# clear environment
rm(list=ls())

library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)

#---- Behavior results----
# Run GLMM on behavior (N = 14)
behavior <- read.csv('/mnt/storage/RASP/results/behavior_stat.csv')

group_behavior <- lmer(behaviorscore ~ Condition + (1 | participant), data = behavior)
# View the summary of the ANOVA
summary(group_behavior)

# estimating pairwise effect size using Cohen's d
library(emmeans)

emm_behavior <- emmeans(group_behavior, ~Condition)
emm_behavior

pairs(emm_behavior)

eff_size(emm_behavior,sigma = sigma(group_behavior),edf = df.residual(group_behavior))

# get means and sd
behavior_summary <- behavior %>%
  group_by(Condition) %>%  # <-- Check this line carefully
  summarise(
    mean_score = mean(behaviorscore, na.rm = TRUE),
    sd_score   = sd(behaviorscore, na.rm = TRUE)
  )


behavior_summary

# Behavior visualization
# Base jitter
pos <- position_jitter(width = 0.1)

ggplot(behavior, aes(x = Condition, y = behaviorscore)) +
  geom_boxplot(fill = "lightblue", alpha = 0.5) +
  
  # All points EXCEPT participants 28 and 62
  geom_point(
    data = subset(behavior, !participant %in% c('rasp_28', 'rasp_62')),
    position = pos,
    size = 2
  ) +
  
  # Participants 28 and 62 in red, using the SAME jitter
  geom_point(
    data = subset(behavior, participant %in% c('rasp_28', 'rasp_62')),
    position = pos,
    color = "red",
    size = 2
  ) +
  
  scale_x_discrete(labels = c("BA" = "BRD", "CT" = "NRD", "TA" = "TRD")) +
  labs(x = "Condition", y = "Percent Correct") +
  theme_minimal()

  
# ---- EEG data analysis with 17 participants -----

# Load a CSV file into R
eeg_ta <- read.csv("/mnt/storage/RASP/results/eeg_stat_ta_gamma_plvs.csv")
eeg_bg <- read.csv("/mnt/storage/RASP/results/eeg_stat_bg_gamma_plvs.csv")
eeg_diff <- read.csv("/mnt/storage/RASP/results/eeg_stat_diff_gamma_plvs.csv")


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

# ggplot(data_long, aes(x = Condition, y = MeanChall, fill = Stream)) +
#  geom_violin(position = position_dodge(width = 0.8), alpha = 0.5) +
#  geom_jitter(aes(color = Stream),
#              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
#              size = 1.5, alpha = 0.7) +
#  labs(title = "Mean PLVs by Condition and Stream",
#       x = "Condition", y = "Mean Across All Channels") +
#  theme_minimal()

## boxplot ##
pos_jd <- position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8)

ggplot(data_long, aes(x = Condition, y = MeanChall, fill = Stream)) +
  geom_boxplot(
    position = position_dodge(width = 0.8),
    alpha = 0.5,
    outlier.shape = NA,
    width = 0.6
  ) +
  
  # All points EXCEPT the ones you want black
  geom_jitter(
    data = subset(data_long, !participant %in% c('rasp_13', 'rasp_37', 'rasp_38', 'rasp_39', 'rasp_50')),
    aes(color = Stream),
    position = pos_jd,
    size = 1.5,
    alpha = 0.7
  ) +
  
  # Highlighted participants in black
  geom_jitter(
    data = subset(data_long, participant %in% c('rasp_13', 'rasp_37', 'rasp_38', 'rasp_39', 'rasp_50')),
    aes(color = NULL),   # remove Stream color mapping
    position = pos_jd,
    color = "black",
    size = 1.8,
    alpha = 0.9
  ) +
  
  scale_x_discrete(labels = c("BA" = "BRD", "CT" = "NRD", "TA" = "TRD")) +
  scale_fill_manual(
    values = c("mean_chall_ta" = "red", "mean_chall_bg" = "blue"),
    labels = c("Background Stream", "Target Stream")
  ) +
  scale_color_manual(
    values = c("mean_chall_ta" = "red", "mean_chall_bg" = "blue"),
    labels = c("Background Stream", "Target Stream")
  ) +
  labs(
    x = "Condition",
    y = "Mean Across All Channels",
    fill = "Stream",
    color = "Stream"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(
    color = "none",
    fill = guide_legend(title = "Stream")
  )

# if only we want to try to use a different condition as referece, default is BA (BRD)
# Ensure Condition is a factor
#data_long$Condition <- factor(data_long$Condition)

# Set "TRD" as the reference level
#data_long$Condition <- relevel(data_long$Condition, ref = "TA")

# Fit the model
group_eeg <- lmer(MeanChall ~ Condition * Stream + (1 | participant), data = data_long)

# Summary of the model
summary(group_eeg)

# pairwise contrasts
em_eeg <- emmeans(group_eeg, ~ Stream) #or Stream | Condition

pairs(em_eeg)

# Compute effect sizes (Cohen's d) for those interaction contrasts 
interaction_effsize <- eff_size(em_eeg, sigma = sigma(group_eeg), edf = df.residual(group_eeg)) 
interaction_effsize

# post hoc tests
posthoc <- pairs(em_eeg, adjust = "tukey")

# View significant results only
summary(posthoc) %>% filter(p.value < 0.05)

## now look at difference (delta) in PLV between target and background stream

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

## boxplot ##
ggplot(data_filtered_diff, aes(x = Condition, y = mean_chall)) +
  geom_boxplot(fill = "lightblue", color = "black", alpha = 0.6, outlier.shape = NA) +
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

#ggplot(data_long, aes(x = Channel['F6','F8','PO3'], y = Value, fill = Condition)) +
#  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5) +
#  geom_jitter(aes(color = Condition),
#              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
#               size = 1.5, alpha = 0.7) +
#   labs(title = "Mean PLVs by Condition and Stream",
#        x = "Condition", y = "Mean Across All Channels") +
#   theme_minimal()

# Fit the model
model <- lmer(Value ~ Condition * Channel + (1 | participant), data = data_long)

# Summary of the model
summary(model)

# Extract summary table
model_summary <- summary(model)

# Get fixed effects table
fixed_effects <- as.data.frame(model_summary$coefficients)

# Filter for significant effects
significant_effects <- fixed_effects[fixed_effects$`Pr(>|t|)` < 0.1, ]

# View results
print(significant_effects)

## plot only the significant channels
library(dplyr)

# Filter only the channels you want
data_subset <- data_long %>%
  filter(Channel %in% c("F6", "F8", "PO3"))

# Plot the significant channels
data_subset$Condition <- factor(data_subset$Condition)

ggplot(data_subset, aes(x = Channel, y = Value, fill = Condition)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5) +
  geom_jitter(aes(color = Condition),
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 1.5, alpha = 0.7) +
  scale_fill_manual(values = c("BA" = "#E69F00",   # Orange
                               "CT" = "#009E73",   # Green
                               "TA" = "#CC79A7"),
                    labels = c('BRD','NRD','TRD')) + # Purple
  scale_color_manual(values = c("BA" = "#E69F00",
                                "CT" = "#009E73",
                                "TA" = "#CC79A7"),
                     labels = c('BRD','NRD','TRD')) +
  labs(y = "Delta PLV") +
  theme_minimal()

ggplot(data_subset, aes(x = Channel, y = Value, fill = Condition, color = Condition)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5, outlier.shape = NA) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8),
              size = 1.5, alpha = 0.7) +
  scale_fill_manual(values = c("BA" = "#E69F00", "CT" = "#009E73", "TA" = "#CC79A7"),
                    labels = c('BRD','NRD','TRD')) +
  scale_color_manual(values = c("BA" = "#E69F00", "CT" = "#009E73", "TA" = "#CC79A7"),
                     labels = c('BRD','NRD','TRD')) +
  labs(y = "Difference in PLV") +
  theme_minimal()

emm <- emmeans(model, ~ Condition | Channel)

# Pairwise comparisons across Condition within each Channel
posthoc <- pairs(emm, adjust = "tukey")

# View significant results only
summary(posthoc) %>% filter(p.value < 0.05)

pairs(emm)

# Compute effect sizes (Cohen's d) for those interaction contrasts 
interaction_effsize <- eff_size(emm, sigma = sigma(model), edf = df.residual(model)) 
interaction_effsize

## now analyze EEG along with behavior

# ---- Subset of 12 people with behavior and EEG----

# Load a CSV file into R
merged_diff <- read.csv('/mnt/storage/RASP/results/merged_stat_diff_gamma_plvs.csv')

# View the first few rows
head(merged_diff)

# add the significant cluster
sigclu_diff <- c('F6')   # based on previous model output
merged_diff$mean_sigclu <- merged_diff[, sigclu_diff]

head(merged_diff$mean_sigclu)

#visualize PLV results just at F6
ggplot(merged_diff, aes(x = Condition, y = mean_sigclu)) +
  geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.5) +
  geom_point(aes(group = participant), position = position_jitter(width = 0.1), size = 2) +
  geom_line(aes(group = participant), color = "gray", alpha = 0.5) +
  labs(title = "Difference in PLVs by Condition",
       x = "Condition", y = "Target - Background PLV") +
  theme_minimal()

# Run multiple regression with interaction
model_diff <- lmer(behaviorscore ~ mean_sigclu + (1 | participant), data = merged_diff)

# View the summary
summary(model_diff)

# getting R2 as effect size
library(MuMIn)
r.squaredGLMM(model_diff)

library(performance)
r2(model_diff)

# visualize the model
ggplot(merged_diff, aes(x = mean_sigclu, y = behaviorscore)) +
  geom_point(color = "black", size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, 
              color = "lightblue", linetype = "dashed", size = 1) +
  labs(x = "Difference in PLV", 
       y = "Behavior Score") +
  theme_minimal()
