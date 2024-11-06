# BREAKING (THE) NEWS

An interactive interface for exploring daily (German) news coverage across **political spectrums**, this project analyzes top news stories, identifies media **blind spots**, assesses **sentiment**, and uncovers **interlinked topics** to offer a comprehensive view of the day’s news landscape.

## Features
- **Top News Stories**: Shows the top news stories of the day and reporting frequency by media outlet.
- **Political Spectrum Analysis**: Identifies the political spectrum of reporting media and highlights potential blind spots.
- **Sentiment Analysis**: Uses NLP to determine the tone and focus of news coverage, helping to reveal bias and sentiment differences across political leanings.
- **Related Topics Visualization**: Connects current topics to show hidden relationships in the news.
- **Historical Perspective**: Analysis of histrical trends in reporting about specific news topics.
  
## Tech Stack
- **Frontend**: Streamlit for the interactive user interface.
- **Backend**: [Newscatcher API](https://www.newscatcherapi.com/), News API, Mediastack API, SQL (Google BigQuery) and Python
- **Data**: GDELT (https://www.gdeltproject.org/), news articles obtained via news API
- **NLP & Sentiment Analysis**: Natural Language Processing techniques and sentiment models (e.g., VADER, BERT Transformers) for tone analysis.

## Planning
**Day 1-2: Securing Data sources**
- APIs for recent Data
- GDELT database connection/queries
- Web scraping articles

**Day 2-3: Set up data pipeline & data cleaning**
- Google BigQuery or Alternative
- Combining recent data and historical data in one database
- Cleaning full article content data (only proper results)

**Day 4-5: Analyzing & model training (e.g. BERT, roBERTa, Gemini)**
- Topic classification
- Related topic score
- Political tendency
- Sentiment

**Day 6-7: Building streamlit interface**
