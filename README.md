# BREAKING THE NEWS

An interactive interface for exploring daily (German) news coverage across political spectrums, this project analyzes top news stories, identifies media blind spots, assesses sentiment, and uncovers interlinked topics to offer a comprehensive view of the day’s news landscape.

## Features
- **Top News Stories**: Shows the top news stories of the day and reporting frequency by media outlet.
- **Political Spectrum Analysis**: Identifies the political spectrum of reporting media and highlights potential blind spots.
- **Sentiment Analysis**: Uses NLP to determine the tone and focus of news coverage, helping to reveal bias and sentiment differences across political leanings.
- **Related Topics Visualization**: Connects current topics to show hidden relationships in the news.
- **Historical Perspective**: Analysis of histrical trends in reporting about specific news topics.
  
## Tech Stack
- **Frontend**: Streamlit for the interactive user interface.
- **Backend & Data**: News API, Mediastack API, GDELT (for historical data), SQL (Google BigQuery) and Python.
- **NLP & Sentiment Analysis**: Natural Language Processing techniques and sentiment models (e.g., VADER, BERT Transformers) for tone analysis.
