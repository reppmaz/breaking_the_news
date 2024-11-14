import streamlit as st
import pandas as pd
from collections import Counter
from ast import literal_eval
import ast
import plotly.graph_objects as go
import plotly.express as px

#--------------
# GENERAL SETUP
#--------------
# Load data
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = df['datetime'].astype(str)
    df['entities'] = df['entities'].apply(literal_eval)
    return df

# load data
data_filepath = "/Users/reppmazc/Documents/IRONHACK/quests/final_project/breaking_news_data.csv"
df = load_data(data_filepath)

# --- Color Mappings ---
pol_leaning_colors = {
    'links': '#fe292b',
    'mitte_links': '#fed26c',
    'mitte': '#feaaac',
    'mitte_rechts': '#2bb19d',
    'rechts': '#036acc'}

# Color for sentiment
sentiment_colors = {
    'positive': '#68b25d',
    'neutral': '#676269',
    'negative': '#d94eda'}

#--------------
# PROJECT TITLE
#--------------
st.title("BREAKING (THE) NEWS")
st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

#-------------------------------
# TOPIC OF THE DAY - Calculation
#-------------------------------
# Select the 300 most recent articles for topic of the day
df_recent = df.dropna(subset=['datetime']).sort_values(by='datetime', ascending=False).head(300)
top_topic = df_recent['topic'].value_counts().idxmax()
top_10_topics = df['topic'].value_counts().nlargest(10).index.tolist()

# Custom CSS for styling
st.markdown("""
    <style>
    .stRadio > div { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
    .stRadio > div label { background-color: #5b5b5c; padding: 8px 12px; border-radius: 20px; color: white; font-weight: bold; cursor: pointer; transition: background-color 0.2s ease; }
    .stRadio > div label:hover { background-color: #036acc; }
    .stRadio > div label[data-selected="true"] { background-color: #036acc; }
    </style>
""", unsafe_allow_html=True)

# Topic selection
st.markdown("<p style='font-size:20px; color:#5b5b5c'><strong>Welches Nachrichtenthema sollen wir für dich analysieren?</strong></p>", unsafe_allow_html=True)
selected_topic = st.radio("", options=top_10_topics, index=top_10_topics.index(top_topic))
st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

# Sample articles selection by political leaning (use all data)
selected_articles = []
for leaning in pol_leaning_colors.keys():
    articles = df[(df['topic'] == selected_topic) & (df['pol_leaning'] == leaning)]
    if not articles.empty:
        selected_articles.append(articles.sample(1))

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Das wird über {selected_topic} gesagt:")
    
    # Display the selected articles with links
    for article in selected_articles:
        st.markdown(f"- [{article.iloc[0]['source']}]({article.iloc[0]['url']}) (eher {article.iloc[0]['pol_leaning']})")
    
    # Calculate percentage of articles by political leaning, including leanings with 0%
    pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
    pol_leaning_counts = pol_leaning_counts.reindex(pol_leaning_colors.keys(), fill_value=0)  # Ensure all leanings are included
    total_articles = pol_leaning_counts.sum()
    pol_leaning_percentages = (pol_leaning_counts / total_articles * 100).round(2)

    # Check for any blind spots (pol_leaning with <= 10%)
    blind_spots = pol_leaning_percentages[pol_leaning_percentages <= 10]
    if not blind_spots.empty:
        st.markdown("<br>", unsafe_allow_html=True)  # Add an empty line
        st.markdown("### Blinder Fleck")
        st.write("Die folgenden politischen Seiten sprechen wenig über dieses Thema:")
        for leaning, pct in blind_spots.items():
            if pct == 0.0:
                st.write(f"- **{leaning}** mit gar keiner Berichterstattung")
            else:
                st.write(f"- **{leaning}** mit nur **{pct}%** der Berichterstattung")

with col2:
    st.subheader(f"Wer berichtet wie viel über {selected_topic}:")

    # Filter out 0 values and prepare data for the chart (use all data)
    pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
    pol_leaning_counts = pol_leaning_counts[pol_leaning_counts > 0]  # Exclude 0 values
    labels, values = pol_leaning_counts.index, pol_leaning_counts.values
    colors = [pol_leaning_colors[leaning] for leaning in labels]

    # Create the Plotly donut chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, marker=dict(colors=colors))])

    # Update layout with adjusted legend position and moderate margins
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.2),  # Minor adjustment to x position
        margin=dict(t=40, b=40, l=40, r=40),  # Slightly larger margins to avoid overlap
        paper_bgcolor="#0e1214",
        font=dict(color="#f8f8fa")
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

# ---------
# SENTIMENT
# ---------
st.subheader(f"So ist die Stimmung zu {selected_topic}:")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<p style='font-size:16px'>Gesamtstimmung über alle Medien:</p>", unsafe_allow_html=True)
    # Calculate sentiment counts without reindexing prematurely
    total_sentiment_counts = df[df['topic'] == selected_topic]['sentiment'].value_counts()
    
    # Only reindex if a sentiment category is missing
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment not in total_sentiment_counts:
            total_sentiment_counts[sentiment] = 0
    total_sentiment_counts = total_sentiment_counts[['negative', 'neutral', 'positive']]

    # Create the bar chart for total sentiment
    fig = go.Figure(data=[
        go.Bar(
            x=total_sentiment_counts.index,
            y=total_sentiment_counts.values,
            marker_color=[sentiment_colors[s] for s in total_sentiment_counts.index])])
    fig.update_layout(
        xaxis_title="Stimmung",
        yaxis_title="Anzahl der Artikel",
        paper_bgcolor="#0e1214",
        plot_bgcolor="#0e1214",
        font=dict(color="#f8f8fa"),
        showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<p style='font-size:16px'>Stimmung nach Medium:</p>", unsafe_allow_html=True)
    # Calculate sentiment counts by source without forcing reindex prematurely
    sentiment_counts = df[df['topic'] == selected_topic].groupby('source')['sentiment'].value_counts().unstack(fill_value=0)
    
    # Ensure all columns (sentiment categories) are included
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0
    sentiment_counts = sentiment_counts[['negative', 'neutral', 'positive']]  # Ensure order of sentiments
    
    # Create a Plotly stacked bar chart for sentiment by source
    fig2 = go.Figure()
    for sentiment in sentiment_counts.columns:
        fig2.add_trace(
            go.Bar(
                name=sentiment,
                x=sentiment_counts.index,
                y=sentiment_counts[sentiment],
                marker_color=sentiment_colors[sentiment]))

    fig2.update_layout(
        xaxis_title="",
        yaxis_title="Anzahl der Artikel",
        barmode="stack",
        paper_bgcolor="#0e1214",
        plot_bgcolor="#0e1214",
        font=dict(color="#f8f8fa"),
        legend=dict(title="Stimmung"))
    st.plotly_chart(fig2, use_container_width=True)

# -------------------
# REPORTING OVER TIME
# -------------------
st.subheader(f"So oft wurde über {selected_topic} berichtet:")

# Filter out rows with NaN, invalid 'datetime' values, or years before 2023
df = df.dropna(subset=['datetime'])
df = df[df['datetime'].str.match(r'^\d{8,}')]
df['Datum'] = df['datetime'].str[:6]

# Filter to keep only rows with years >= 2023
df = df[df['Datum'].str[:4].astype(int) >= 2023]

# Calculate total articles per source to normalize
total_articles_per_source = df[df['topic'] == selected_topic].groupby('source').size()

# Group by 'Datum' and 'source' to count articles and normalize by total count
monthly_topic_data = (
    df[df['topic'] == selected_topic]
    .groupby(['Datum', 'source'])
    .size()
    .div(total_articles_per_source)  # Normalize by total articles per source
    .unstack(fill_value=0))

# Reshape data for plotting
monthly_topic_data = monthly_topic_data.reset_index().melt(id_vars="Datum", var_name="Source", value_name="Relative Häufigkeit")

# Check if monthly_topic_data is empty after grouping
if monthly_topic_data.empty:
    st.write("No data available for plotting after grouping by year and month.")
else:
    # Convert 'Datum' to a datetime object for a continuous x-axis
    monthly_topic_data['Datum'] = pd.to_datetime(monthly_topic_data['Datum'], format='%Y%m')

    # Create the line plot with Plotly
    fig = px.line(monthly_topic_data, x="Datum", y="Relative Häufigkeit", color="Source")

    # Update x-axis to show each month
    fig.update_xaxes(
        tickformat="%b %Y",
        dtick="M1")

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# --------------
# Related Topics
# --------------
related_topics = df.loc[df['topic'] == selected_topic, 'related_topics'].values[0]
if isinstance(related_topics, str):
    related_topics = ast.literal_eval(related_topics)

# Filter related topics to include only those with at least 2 overlapping entities
related_topics = [t for t in related_topics if t[1] >= 2]

# Sort related topics by shared entity count (second element in each tuple) in descending order
related_topics = sorted(related_topics, key=lambda x: x[1], reverse=True)[:5]

# Extract names and counts
related_topic_names = [t[0] for t in related_topics]
shared_entity_counts = [t[1] for t in related_topics]

# Create a heatmap with unique colors for each square and counts displayed inside
fig = go.Figure(data=go.Heatmap(
    z=[shared_entity_counts],
    x=related_topic_names,
    y=[selected_topic],
    colorscale='BuPu',
    showscale=False,
    text=[shared_entity_counts],  # Display counts inside each square
    texttemplate="%{text}",  # Show text in each square
    textfont=dict(color="white")  # Make text color white for contrast
))

# Customize layout
fig.update_layout(
    title=f"Gemeinsame Entitäten zwischen {selected_topic} und verwandten Themen",
    xaxis=dict(tickangle=45),
    yaxis=dict(),
    paper_bgcolor="#0e1214",
    plot_bgcolor="#0e1214",
    font=dict(color="#f8f8fa")
)

# Display the heatmap in Streamlit
st.plotly_chart(fig, use_container_width=True)

#--------------
# ABOUT SECTION
#--------------
st.markdown("<p style='font-size:20px; color:#5b5b5c'><strong>Über das Projekt</strong></p>", unsafe_allow_html=True)

with st.expander("Projektinformationen anzeigen"):
    st.markdown("""
        Diese Informationen dienen der Transparenz und Nachvollziehbarkeit unserer Analyseprozesse. 

        **Datenquellen**: Diese Anwendung verwendet deutsche Nachrichtenartikel der letzten 12 Monate:
        - vom Gdelt Project: https://www.gdeltproject.org/
        - und gesammelt über news APIs: NewsCatcher, NewsAPI, MediaStack
        
        **Methodik**
        - **Themen Klassifizierung**: paraphrase-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        - **Politische Einordnung**: Medien wurden basierend auf 4 Quellen auf dem politischen Spektrum verordnet:
            - https://interaktiv.tagesspiegel.de/lab/die-lieblingsmedien-der-parteien/
            - https://uebermedien.de/93000/wieso-haben-zeitungen-eine-politische-ausrichtung/
            - https://www.pewresearch.org/global/fact-sheet/datenblatt-nachrichtenmedien-und-politische-haltungen-in-deutschland/
            - https://www.polkom.ifp.uni-mainz.de/files/2024/01/pm_perspektivenvielfalt.pdf
        - **Blinderfleck-Analyse**: definiert als politische Ausrichtungen mit ≤10% Anteil an der Berichterstattung zum ausgewählten Thema
        - **Stimmungs-Analyse**: XLM-RoBERTa-German-sentiment https://huggingface.co/ssary/XLM-RoBERTa-German-sentiment
        - **Verwandte Themen**: zeigt die 5 Themen mit den meisten (mind. 2) gemeinsamen Entitäten
    """)
