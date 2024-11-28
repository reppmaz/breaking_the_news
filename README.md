# Breaking (The) News

An interactive dashboard for exploring daily German news across various political perspectives. The platform analyzes top stories, media blind spots, sentiment, and interrelated topics to offer a comprehensive view of the day's news landscape.

## Project Overview

In an era dominated by media bubbles, "Breaking (The) News" enhances media literacy by revealing biases and connections in news. Using data from multiple sources, the project applies machine learning to help users explore current events and historical trends with transparency.

## Key Features

- **Top News Stories**: Displays top stories and reporting frequency by source.
- **Political Spectrum Analysis**: Maps stories across the political spectrum, highlighting blind spots.
- **Sentiment Analysis**: NLP-based tone and bias detection across outlets.
- **Related Topics Visualization**: Connects trending topics for deeper insights.
- **Historical Perspective**: Tracks changes in topic reporting over time.

## Tech Stack

- **Frontend**: Streamlit for the user interface.
- **Backend**: Google BigQuery, Newscatcher API, News API, Mediastack API.
- **Data Sources**: GDELT for historical data; APIs and web scraping for real-time articles.
- **NLP**: VADER, BERT, and RoBERTa for sentiment and topic modeling.

## Data Workflow

1. **Data Sourcing**: API calls (~300/day), GDELT queries, and web scraping (~38,000 articles).
2. **Data Cleaning**: Language filtering, deduplication, removal of irrelevant text and symbols.
3. **Analysis**: Sentiment analysis, topic modeling, and political spectrum classification.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/reppmaz/breaking_the_news.git

## Project Links

[Presentation](https://docs.google.com/presentation/d/1lxwm64N-_AGfRvqF-qgQeYougpIizXZ4liRcaKDkHQo/edit?usp=sharing)
