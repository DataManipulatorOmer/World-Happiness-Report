import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
def main():
    st.title("World Happiness Report Data Exploration")
    fileUploaded = st.file_uploader("Choose the World Happiness Report CSV file", type=["csv"])

    if fileUploaded is not None:
        df = pd.read_csv(fileUploaded)
        st.header("General Information about the World Happiness Report Dataset:")
        st.write(df.info())

        # Display summary statistics
        st.header("Summary Statistics of the Dataset:")
        st.write(df.describe())

        # Explore the top and bottom countries in terms of happiness scores
        st.header("Top Countries in Terms of Happiness Scores:")
        topCount = df.nlargest(15, 'Score')  # Change 5 to any desired number
        st.table(topCount[['Country or region', 'Score']])

        st.header("Bottom Countries in Terms of Happiness Scores:")
        bottomCount = df.nsmallest(5, 'Score')  # Change 5 to any desired number
        st.table(bottomCount[['Country or region', 'Score']])
        
                # Visualization 1: Bar chart of happiness scores by country
        # Visualization: Bar chart of happiness scores for the top 10 countries
        st.header("Bar Chart: Top 10 Countries with Highest Happiness Scores")
        top10 = df.nlargest(10, 'Score')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top10['Country or region'], top10['Score'], color='skyblue')
        ax.set_xticklabels(top10['Country or region'], rotation=45, ha='right')
        ax.set_ylabel('Happiness Score')
        st.pyplot(fig)
        # Visualization 2: World map with color-coded happiness scores
        st.header("World Map: Happiness Scores by Country")
        Mapp = px.choropleth(df, 
                                locations='Country or region', 
                                locationmode='country names',
                                color='Score',
                                color_continuous_scale='Viridis',
                                title='Happiness Scores by Country',
                                labels={'Score': 'Happiness Score'})
        st.plotly_chart(Mapp)
        
        st.header("Overall Rank")
                # Check if the dataset includes multiple years
        if 'Overall rank' in df.columns:
            # Visualization: Line chart of average happiness scores per year
            st.header("Line Chart: Average Happiness Scores Per Year")
            yearlyScore = df.groupby('Overall rank')['Score'].mean().reset_index()
            avgLine = px.line(yearlyScore, x='Overall rank', y='Score', title='Average Happiness Scores Per Year')
            st.plotly_chart(avgLine)

        else:
            st.warning("The dataset does not include information about multiple years.")
            
        
        # Visualization: Scatter plot of GDP per capita vs Happiness Score
        st.header("Scatter Plot: GDP per Capita vs Happiness Score")
        gdpViz = px.scatter(df, x='GDP per capita', y='Score', title='GDP per Capita vs Happiness Score')
        st.plotly_chart(gdpViz)

        # Visualization: Scatter plot of Social Support vs Happiness Score
        st.header("Scatter Plot: Social Support vs Happiness Score")
        ssViz = px.scatter(df, x='Social support', y='Score', title='Social Support vs Happiness Score')
        st.plotly_chart(ssViz)

        # Visualization: Scatter plot of Healthy Life Expectancy vs Happiness Score
        st.header("Scatter Plot: Healthy Life Expectancy vs Happiness Score")
        healthViz = px.scatter(df, x='Healthy life expectancy', y='Score', title='Healthy Life Expectancy vs Happiness Score')
        st.plotly_chart(healthViz)

        # Visualization: Bar chart of the impact of Freedom on Happiness Score
        st.header("Bar Chart: Impact of Freedom on Happiness Score")
        freedomViz = px.bar(df, x='Country or region', y='Freedom to make life choices', title='Impact of Freedom on Happiness Score')
        freedomViz.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(freedomViz)

        # Visualization: Bar chart of Generosity impact on Happiness Score
        st.header("Bar Chart: Impact of Generosity on Happiness Score")
        generosityViz = px.bar(df, x='Country or region', y='Generosity', title='Impact of Generosity on Happiness Score')
        generosityViz.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(generosityViz)

        # Visualization: Bar chart of Perceptions of Corruption impact on Happiness Score
        st.header("Bar Chart: Impact of Perceptions of Corruption on Happiness Score")
        corruptionViz = px.bar(df, x='Country or region', y='Perceptions of corruption', title='Impact of Perceptions of Corruption on Happiness Score')
        corruptionViz.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(corruptionViz)
        
        # Check if the dataset includes the required columns
        colCheck = ['Overall rank', 'Country or region', 'Score', 'GDP per capita', 'Social support',
                             'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
        if set(colCheck).issubset(df.columns):
            # Sidebar widgets for user interaction
            yearSel = st.sidebar.slider("Select a Year", min_value=int(df['Overall rank'].min()), max_value=int(df['Overall rank'].max()))
            regionSel = st.sidebar.selectbox("Select a Region", df['Country or region'].unique())
            countrySel = st.sidebar.selectbox("Select a Country", df['Country or region'].unique())

            # Filter the DataFrame based on user selections
            dataFiltered = df[df['Overall rank'] == yearSel]

            # Display the filtered data
            st.header("Filtered Data:")
            st.dataframe(dataFiltered)

            # Visualization: Scatter plot of GDP per capita vs Happiness Score
            st.header("Scatter Plot: GDP per Capita vs Happiness Score")
            gdpViz = px.scatter(dataFiltered, x='GDP per capita', y='Score', title='GDP per Capita vs Happiness Score')
            st.plotly_chart(gdpViz)

            # Additional visualizations or analysis based on user selection can be added here

        else:
            st.warning("The dataset does not include all required columns.")
            
        colCheck = ['Overall rank', 'Country or region', 'Score', 'GDP per capita', 'Social support',
                             'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
        if set(colCheck).issubset(df.columns):
            # Sidebar widgets for user interaction
            selected_feature = st.sidebar.selectbox("Select a Feature for Prediction", df.columns[3:])  # Assuming you want to predict 'Score' based on other features

            # Split the data into training and testing sets
            X = df.drop(['Score', 'Overall rank', 'Country or region'], axis=1)  # Features excluding 'Score', 'Overall rank', 'Country or region'
            y = df['Score']  # Target variable 'Score'

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display model evaluation metrics
            st.header("Linear Regression Model Evaluation:")
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R-squared: {r2}")

            # Visualization: Actual vs Predicted Happiness Scores
            st.header("Scatter Plot: Actual vs Predicted Happiness Scores")
            fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Score', 'y': 'Predicted Score'})
            st.plotly_chart(fig_scatter)

        else:
            st.warning("The dataset does not include all required columns.")
if __name__ == "__main__":
    main()
