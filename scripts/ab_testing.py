import pandas as pd
import scipy.stats as stats

def calculate_risk_ratio(data):
    # Calculate the Risk Ratio (TotalClaims / TotalPremium)
    data['RiskRatio'] = data['TotalClaims'] / data['TotalPremium']

def calculate_profit_margin(data):
    # Calculate Profit Margin
    data['ProfitMargin'] = data['TotalPremium'] - data['TotalClaims']
    # Remove NaN values from ProfitMargin
    data = data.dropna(subset=['ProfitMargin'])
    return data

# Categorical Data Testing: Chi-Squared Test
def chi_squared_test(df, feature, target):
    # Create a contingency table
    contingency_table = pd.crosstab(df[feature], df[target])
    # Perform chi-squared test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_value

# # Numerical Data Testing: T-Test (for two groups) and ANOVA (for multiple groups)
# def t_test(df, group1, group2, numerical_feature):
#     # Filter data by groups
#     group1_data = df[df[group1] == 'Group1'][numerical_feature]
#     group2_data = df[df[group2] == 'Group2'][numerical_feature]
#     # Perform t-test
#     t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
#     return t_stat, p_value

def t_test(df, gender_column, group1, group2, numerical_feature):
     # Filter out 'Not specified' and drop NaN RiskRatio entries
    df_filtered = df[df[gender_column].isin([group1, group2]) & df[numerical_feature].notna()]
    
    # Filter data by groups
    group1_data = df_filtered[df_filtered[gender_column] == group1][numerical_feature]
    group2_data = df_filtered[df_filtered[gender_column] == group2][numerical_feature]
   
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    return t_stat, p_value

def anova_test(df, numerical_feature, group_feature):
    # Prepare data for ANOVA
    group_data = [group[numerical_feature].dropna() for name, group in df.groupby(group_feature)]
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*group_data)
    return f_stat, p_value
