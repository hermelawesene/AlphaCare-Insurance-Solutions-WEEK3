import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Stat():
    def __init__(self, data):
        self.data = data

    def decriptive_stat(self,columns):
        stats = {}
        for col in columns:
            stats[col] = {
                "Mean": self.data[col].mean(),
                "Standard Deviation": self.data[col].std(),
                "Variance": self.data[col].var(),
                "Range": self.data[col].max() - self.data[col].min(),
                "Interquartile Range (IQR)": self.data[col].quantile(0.75) - self.data[col].quantile(0.25)
            }
        return stats
    
    def review_data_structure(self):
        # Review data types
        print("Data Types of Each Column:")
        print(self.data.dtypes)
        
        # Check for categorical variables
        print("\nPotential Categorical Variables:")
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        print(categorical_cols)
        
        # Check for columns that could be DateTime
        print("\nColumns that could be DateTime:")
        potential_date_cols = []
        for col in self.data.columns:
            try:
                # Try converting column to datetime with a format if possible
                # If you know the expected format (e.g., 'YYYY-MM-DD'), set it using the format parameter
                converted_col = pd.to_datetime(self.data[col], errors='coerce', format='%Y-%m-%d')  # Adjust format as needed
                if converted_col.notnull().sum() > 0:  # Ensure there are valid dates after conversion
                    potential_date_cols.append(col)
                    print(f"{col} could be DateTime.")
            except (ValueError, TypeError):
                continue
        
        if potential_date_cols:
            print(f"\nPotential DateTime columns: {potential_date_cols}")
        else:
            print("No potential DateTime columns detected.")

    def check_missing_values(self):
        # Check for missing values
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            print("\nMissing Values Summary:")
            print(missing_data)
        else:
            print("\nNo missing values found.")

    def plot_histograms(self, numerical_cols):
        """Plot histograms for numerical columns."""
        for col in numerical_cols:
            # Calculate median of the column
            median_value = self.data[col].median()
        
            # Fill missing values with the median
            self.data[col] = self.data[col].fillna(median_value)
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data[col], bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def plot_bar_charts(self, categorical_cols):
        """Plot bar charts for categorical columns."""
        for col in categorical_cols:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.data, x=col, order=self.data[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.show()

    def plot_scatter_plots(self, x_col, y_col, hue_col):
        """Plot scatter plots of two variables with hue as a categorical variable."""
        relevant_columns = ['TotalPremium', 'TotalClaims', 'PostalCode']
        df_relevant = self.data[relevant_columns].dropna()
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_relevant, x=x_col, y=y_col, hue=hue_col, palette='viridis', alpha=0.6)
        plt.title(f"Scatter Plot of {x_col} vs {y_col} by {hue_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(title=hue_col)
        plt.show()

    def plot_correlation_matrix(self):
        """Plot a correlation matrix for numerical columns."""
        relevant_columns = ['TotalPremium', 'TotalClaims', 'PostalCode']
        df_relevant = self.data[relevant_columns].dropna()
        corr_matrix = df_relevant.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_geographic_trends(self, df, geographic_column, numerical_column, categorical_column):
        """Plot geographic trends with numerical and categorical data."""
        plt.figure(figsize=(18, 12), dpi=100)

        # Ensure columns are treated as strings for plotting
        df[geographic_column] = df[geographic_column].astype(str)
        df[categorical_column] = df[categorical_column].astype(str)

        # Plot trends for numerical columns
        plt.subplot(2, 1, 1)
        sns.boxplot(x=geographic_column, y=numerical_column, data=df)
        plt.title(f"Trends of {numerical_column} by {geographic_column}")
        plt.xticks(rotation=45)

        # Plotting trends for categorical columns
        plt.subplot(2, 1, 2)
        sns.countplot(x=categorical_column, data=df, hue=geographic_column)
        plt.title(f"{categorical_column} Count by {geographic_column}")
        plt.xticks(rotation=45)

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
        plt.show()

    def plot_box_plots(self, numerical_columns):
        """Plot box plots to detect outliers in numerical columns."""
        plt.figure(figsize=(14, 8))
        
        for i, col in enumerate(numerical_columns, start=1):
            plt.subplot(len(numerical_columns), 1, i)
            sns.boxplot(y=self.data[col])
            plt.title(f'Box Plot of {col}')
        
        plt.tight_layout()
        plt.show()

    