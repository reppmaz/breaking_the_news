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
#data_filepath = "/Users/reppmazc/Documents/IRONHACK/quests/final_project/breaking_news_data.csv"
#data_filepath = "/Users/reppmazc/Documents/IRONHACK/quests/final_project/combined_file_new.csv"
#data_filepath = "/Users/reppmazc/Documents/IRONHACK/quests/final_project/breaking_the_news/breaking_news_data_new.csv"
data_filepath = "breaking_news_data_new.csv"

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

# Create tabs for Analysis and About sections
tab1, tab2 = st.tabs(["Politische Analyse", "Über das Projekt"])

# Political Analysis Tab
with tab1:
    #--------------
    # PROJECT TITLE
    #--------------
    st.title("BREAKING (THE) NEWS")
    st.markdown("<hr style='border:2px solid #FFF'>", unsafe_allow_html=True)

    #-------------------------------
    # TOPIC OF THE DAY - Calculation
    #-------------------------------
    # Select the 300 most recent articles for the topic of the day
    df_recent = df.dropna(subset=['datetime']).sort_values(by='datetime', ascending=False).head(300)
    top_topic = df_recent['topic'].value_counts().idxmax()  # Most frequent topic in recent articles

    # Get the top 10 topics overall
    top_10_topics = df['topic'].value_counts().nlargest(10).index.tolist()

    # Handle cases where top_topic is not in top_10_topics
    if top_topic not in top_10_topics:
        top_10_topics.insert(0, top_topic)  # Add it to the top of the list

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stRadio > div { display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }
        .stRadio > div label { background-color: #5b5b5c; padding: 8px 12px; border-radius: 30px; color: white; font-weight: bold; cursor: pointer; transition: background-color 0.2s ease; }
        .stRadio > div label:hover { background-color: #036acc; }
        .stRadio > div label[data-selected="true"] { background-color: #036acc; }
        </style>
    """, unsafe_allow_html=True)

    # Topic selection
    st.markdown("<p style='font-size:30px; color:#5b5b5c'><strong>Welches Nachrichtenthema sollen wir für dich analysieren?</strong></p>", unsafe_allow_html=True)
    selected_topic = st.radio("", options=top_10_topics, index=top_10_topics.index(top_topic))
    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Sample articles selection by political leaning (most recent)
    selected_articles = []
    for leaning in pol_leaning_colors.keys():
        # Filter articles for the selected topic and political leaning
        articles = df[(df['topic'] == selected_topic) & (df['pol_leaning'] == leaning)]
        if not articles.empty:
            # Sort articles by datetime in descending order and select the most recent one
            most_recent_article = articles.sort_values(by='datetime', ascending=False).iloc[0]
            selected_articles.append(most_recent_article)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Das wird über {selected_topic} gesagt:")


        # Show selected articles with formatted links
        formatted_articles = ""
        for article in selected_articles:
            formatted_articles += f"""
                <p style='font-size:30px'>- <a href='{article["url"]}' target='_blank' style='color:white;'>{article["source"]}</a> (eher {article["pol_leaning"]})</p>
            """

        # Display the formatted articles
        st.markdown(formatted_articles, unsafe_allow_html=True)

        # percentage of articles by political leaning
        pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
        pol_leaning_counts = pol_leaning_counts.reindex(pol_leaning_colors.keys(), fill_value=0)  # Ensure all leanings are included
        total_articles = pol_leaning_counts.sum()
        pol_leaning_percentages = (pol_leaning_counts / total_articles * 100).round(2)


        # Blind spots (pol_leaning with <= 10%)
        blind_spots = pol_leaning_percentages[pol_leaning_percentages <= 10]
        if not blind_spots.empty:
            st.markdown("<br>", unsafe_allow_html=True)  # Empty line
            st.markdown("### Blinder Fleck")
            st.markdown("""<p style="font-size:30px;">Die folgenden politischen Seiten sprechen wenig über dieses Thema:</p>""", unsafe_allow_html=True)

            # Format blind spots with HTML
            formatted_blind_spots = ""
            for leaning, pct in blind_spots.items():
                if pct == 0.0:
                    formatted_blind_spots += f"""
                        <p style="font-size:30px; line-height:1.5;">- <b>{leaning}</b> mit gar keiner Berichterstattung</p>
                    """
                else:
                    formatted_blind_spots += f"""
                        <p style="font-size:30px; line-height:1.5;">- <b>{leaning}</b> mit nur <b>{pct}%</b> der Berichterstattung</p>
                    """

            # Display the formatted blind spots
            st.markdown(formatted_blind_spots, unsafe_allow_html=True)


    with col2:
        st.subheader(f"Wer berichtet wie viel über {selected_topic}:")

        # filter out 0 values
        pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
        pol_leaning_counts = pol_leaning_counts[pol_leaning_counts > 0]  # Exclude 0 values
        labels, values = pol_leaning_counts.index, pol_leaning_counts.values
        colors = [pol_leaning_colors[leaning] for leaning in labels]

        # donut chart
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker=dict(colors=colors),
                textinfo='percent',
                textfont=dict(size=20)
            )
        ])

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.2,
                font=dict(size=30)
            ),
            margin=dict(t=40, b=40, l=40, r=40),
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
        st.markdown(f"<p style='font-size:30px;'>Gesamtstimmung über alle Medien:</p>", unsafe_allow_html=True)
        # sentiment counts without reindexing prematurely
        total_sentiment_counts = df[df['topic'] == selected_topic]['sentiment'].value_counts()
        
        for sentiment in ['negative', 'neutral', 'positive']:
            if sentiment not in total_sentiment_counts:
                total_sentiment_counts[sentiment] = 0
        total_sentiment_counts = total_sentiment_counts[['negative', 'neutral', 'positive']]

        # bar chart for total sentiment
        fig = go.Figure(data=[
            go.Bar(
                x=total_sentiment_counts.index,
                y=total_sentiment_counts.values,
                marker_color=[sentiment_colors[s] for s in total_sentiment_counts.index])])
        fig.update_layout(
            xaxis=dict(
                title="",
                titlefont=dict(size=30),
                tickfont=dict(size=30)),
                yaxis=dict(
                title="Anzahl der Artikel",
                titlefont=dict(size=30)),
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)



    with col2:
        st.markdown(f"<p style='font-size:30px;'>Stimmung nach Medium:</p>", unsafe_allow_html=True)
        
        # sentiment counts by source
        sentiment_counts = df[df['topic'] == selected_topic].groupby('source')['sentiment'].value_counts().unstack(fill_value=0)
        
        # Ensure all columns (sentiment categories) are included
        for sentiment in ['negative', 'neutral', 'positive']:
            if sentiment not in sentiment_counts.columns:
                sentiment_counts[sentiment] = 0
        sentiment_counts = sentiment_counts[['negative', 'neutral', 'positive']]
        
       # Stacked bar chart for sentiment by source
        fig2 = go.Figure()
        for sentiment in sentiment_counts.columns:
            fig2.add_trace(
                go.Bar(
                    name=sentiment,
                    x=sentiment_counts.index,
                    y=sentiment_counts[sentiment],
                    marker_color=sentiment_colors[sentiment]))

        fig2.update_layout(xaxis=dict(title="",
                                      tickfont=dict(size=30)),
                                      yaxis=dict(title="Anzahl der Artikel",
                                                 titlefont=dict(size=30),
                                                 tickfont=dict(size=30)),
                                                 barmode="stack",
                                                 paper_bgcolor="#0e1214",
                                                 plot_bgcolor="#0e1214",
                                                 font=dict(color="#f8f8fa"),
                                                 legend=dict(title="",
                                                             font=dict(size=30)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # -------------------
    # REPORTING OVER TIME
    # -------------------
    st.subheader(f"So oft wurde über {selected_topic} berichtet:")

    # filter out rows with NaN, invalid 'datetime' values, or years before 2023
    df = df.dropna(subset=['datetime'])
    df = df[df['datetime'].str.match(r'^\d{8,}')]
    df['Datum'] = df['datetime'].str[:6]

    # keep only rows with years >= 2023
    df = df[df['Datum'].str[:4].astype(int) >= 2023]

    # total articles per source for normalization
    total_articles_per_source = df[df['topic'] == selected_topic].groupby('source').size()

    # normalize by total count(per sources)
    monthly_topic_data = (
        df[df['topic'] == selected_topic]
        .groupby(['Datum', 'source'])
        .size()
        .div(total_articles_per_source)  # Normalize by total articles per source
        .unstack(fill_value=0))

    monthly_topic_data = monthly_topic_data.reset_index().melt(id_vars="Datum", var_name="Source", value_name="Relative Häufigkeit")

    if monthly_topic_data.empty:
        st.write("Keine Daten Verfügbar.")
    else:
        monthly_topic_data['Datum'] = pd.to_datetime(monthly_topic_data['Datum'], format='%Y%m')

        # Create line chart
        fig = px.line(monthly_topic_data, x="Datum", y="Relative Häufigkeit", color="Source")

        # Update axes and layout
        fig.update_xaxes(
            title=dict(text="Datum", font=dict(size=30)),
            tickfont=dict(size=30),
            tickformat="%b %Y",
            dtick="M1")
        fig.update_yaxes(title=dict(text="Relative Häufigkeit",
                                    font=dict(size=30)),
                                    tickfont=dict(size=30))
        fig.update_layout(legend=dict(title=dict(text="Source", font=dict(size=30)), 
                font=dict(size=30)),
                paper_bgcolor="#0e1214",
                plot_bgcolor="#0e1214",
                font=dict(color="#f8f8fa"))

        st.plotly_chart(fig, use_container_width=True)


    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # --------------
    # RELATED TOPICS
    # --------------
    st.subheader("Verwandte Themen")

    if 'related_topic_selected' not in st.session_state:
        st.session_state['related_topic_selected'] = selected_topic

    # find related topics
    related_topics = df.loc[df['topic'] == selected_topic, 'related_topics'].values[0]
    if isinstance(related_topics, str):
        related_topics = ast.literal_eval(related_topics)

    # filter related topics to include > 2 overlapping entities
    related_topics = [t for t in related_topics if t[1] >= 2]

    # sort related topics by shared entity count
    related_topics = sorted(related_topics, key=lambda x: x[1], reverse=True)[:5]

    # button styling
    st.markdown("""
        <style>
        .custom-button-container { 
            display: flex; 
            gap: 10px; 
            justify-content: center; 
            flex-wrap: wrap; 
        }
        .stButton>button {
            background-color: #5b5b5c;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 30px;
            border-radius: 30px;
            transition: background-color 0.2s ease;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #036acc;
        }
        </style>
    """, unsafe_allow_html=True)

    # buttons for related topics
    st.markdown("<div class='custom-button-container'>", unsafe_allow_html=True)
    for topic_name, entity_count in related_topics:
        if st.button(f"{topic_name} (Gemeinsame Entitäten: {entity_count})"):
            st.session_state['related_topic_selected'] = topic_name
    st.markdown("</div>", unsafe_allow_html=True)

    # ppdate selected_topic based on user selection
    selected_topic = st.session_state['related_topic_selected']

    st.markdown("<hr style='border:2px solid #FFF'>", unsafe_allow_html=True)
    st.subheader(f"Analysen für das Thema: {selected_topic}")
    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Font size settings for all plots
    FONT_SIZE = 30

    # -------------------------------------
    # Analysis Code for the Selected Topic
    # -------------------------------------

    # Sample Articles Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Das wird über {selected_topic} gesagt:")
        
        # Initialize formatted HTML for displaying articles
        formatted_articles = ""

        for leaning in pol_leaning_colors.keys():
            # Filter articles for the selected topic and political leaning
            articles = df[(df['topic'] == selected_topic) & (df['pol_leaning'] == leaning)]
            if not articles.empty:
                # Select the most recent article
                most_recent_article = articles.sort_values(by='datetime', ascending=False).iloc[0]

                # Add formatted article to the string
                formatted_articles += f"""
                    <p style='font-size:30px'>- <a href='{most_recent_article["url"]}' target='_blank' style='color:white;'>{most_recent_article["source"]}</a> (eher {most_recent_article["pol_leaning"]})</p>
                """
        
        # Display formatted articles
        st.markdown(formatted_articles, unsafe_allow_html=True)



        # Blind Spot Analysis
        pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
        pol_leaning_counts = pol_leaning_counts.reindex(pol_leaning_colors.keys(), fill_value=0)
        total_articles = pol_leaning_counts.sum()
        pol_leaning_percentages = (pol_leaning_counts / total_articles * 100).round(2)

        blind_spots = pol_leaning_percentages[pol_leaning_percentages <= 10]
        if not blind_spots.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Blinder Fleck")
            st.markdown(
                f"<p style='font-size:30px;'>Die folgenden politischen Seiten sprechen wenig über dieses Thema:</p>",
                unsafe_allow_html=True)

            # Initialize formatted text for blind spots
            formatted_blind_spots = ""
            for leaning, pct in blind_spots.items():
                if pct == 0.0:
                    formatted_blind_spots += f"""
                        <p style='font-size:30px;'>- <b>{leaning}</b> mit gar keiner Berichterstattung</p>
                    """
                else:
                    formatted_blind_spots += f"""
                        <p style='font-size:30px;'>- <b>{leaning}</b> mit nur <b>{pct}%</b> der Berichterstattung</p>
                    """

            # Display formatted blind spots
            st.markdown(formatted_blind_spots, unsafe_allow_html=True)


    with col2:
        # Political Leaning Distribution for Selected Topic
        st.subheader(f"Wer berichtet wie viel über {selected_topic}:")
        pol_leaning_counts = pol_leaning_counts[pol_leaning_counts > 0]
        labels, values = pol_leaning_counts.index, pol_leaning_counts.values
        colors = [pol_leaning_colors[leaning] for leaning in labels]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, marker=dict(colors=colors), textfont=dict(size=30))])
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.2,
                font=dict(size=19)),
            margin=dict(t=40, b=40, l=40, r=40),
            paper_bgcolor="#0e1214",
            font=dict(color="#f8f8fa", size=30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)


    # Sentiment Analysis for Selected Topic
    st.subheader(f"So ist die Stimmung zu {selected_topic}:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p style='font-size:30px;'>Gesamtstimmung über alle Medien:</p>", unsafe_allow_html=True)
        total_sentiment_counts = df[df['topic'] == selected_topic]['sentiment'].value_counts()
        for sentiment in ['negative', 'neutral', 'positive']:
            if sentiment not in total_sentiment_counts:
                total_sentiment_counts[sentiment] = 0

        fig = go.Figure(data=[go.Bar(
            x=total_sentiment_counts.index,
            y=total_sentiment_counts.values,
            marker_color=[sentiment_colors[s] for s in total_sentiment_counts.index])])
        fig.update_layout(
            xaxis=dict(title="", titlefont=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
            yaxis=dict(title="Anzahl der Artikel", titlefont=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"<p style='font-size:30px;'>Stimmung nach Medium:</p>", unsafe_allow_html=True)
        sentiment_counts = df[df['topic'] == selected_topic].groupby('source')['sentiment'].value_counts().unstack(fill_value=0)
        for sentiment in ['negative', 'neutral', 'positive']:
            if sentiment not in sentiment_counts.columns:
                sentiment_counts[sentiment] = 0
        sentiment_counts = sentiment_counts[['negative', 'neutral', 'positive']]

        fig2 = go.Figure()
        for sentiment in sentiment_counts.columns:
            fig2.add_trace(
                go.Bar(
                    name=sentiment,
                    x=sentiment_counts.index,
                    y=sentiment_counts[sentiment],
                    marker_color=sentiment_colors[sentiment]))
        fig2.update_layout(
            xaxis=dict(title="", tickfont=dict(size=FONT_SIZE)),
            yaxis=dict(title="Anzahl der Artikel", titlefont=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
            barmode="stack",
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"),
            legend=dict(title="", font=dict(size=FONT_SIZE)))
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Reporting Over Time
    st.subheader(f"So oft wurde über {selected_topic} berichtet:")
    df_filtered = df.dropna(subset=['datetime'])
    df_filtered = df_filtered[df_filtered['datetime'].str.match(r'^\d{8,}')]
    df_filtered['Datum'] = df_filtered['datetime'].str[:6]
    df_filtered = df_filtered[df_filtered['Datum'].str[:4].astype(int) >= 2023]

    total_articles_per_source = df_filtered[df_filtered['topic'] == selected_topic].groupby('source').size()
    monthly_topic_data = (
        df_filtered[df_filtered['topic'] == selected_topic]
        .groupby(['Datum', 'source'])
        .size()
        .div(total_articles_per_source)
        .unstack(fill_value=0))

    if not monthly_topic_data.empty:
        monthly_topic_data = monthly_topic_data.reset_index().melt(
            id_vars="Datum", var_name="Source", value_name="Relative Häufigkeit")
        monthly_topic_data['Datum'] = pd.to_datetime(monthly_topic_data['Datum'], format='%Y%m')

        fig = px.line(monthly_topic_data, x="Datum", y="Relative Häufigkeit", color="Source")
        fig.update_xaxes(tickfont=dict(size=FONT_SIZE), titlefont=dict(size=FONT_SIZE), tickformat="%b %Y", dtick="M1")
        fig.update_yaxes(tickfont=dict(size=FONT_SIZE), titlefont=dict(size=FONT_SIZE))
        fig.update_layout(
            legend=dict(font=dict(size=FONT_SIZE)),
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa", size=FONT_SIZE))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available for plotting after grouping by year and month.")

with tab2:
    # --------------
    # ABOUT SECTION
    # --------------
    # Add main section header
    st.title("BREAKING (THE) NEWS")
    st.markdown("<hr style='border:2px solid #FFF'>", unsafe_allow_html=True)

    # Brief Introduction
    st.markdown("""
        <p style='font-size:30px'>Diese Informationen dienen der Transparenz und Nachvollziehbarkeit unserer Analyseprozesse.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Section: Datenquellen
    st.header("Datenquellen")
    st.markdown("<hr style='border:2px solid #5b5b5c'>", unsafe_allow_html=True)  # Section divider
    st.markdown("""
        <p style='font-size:30px'>Diese Anwendung verwendet deutsche Nachrichtenartikel der letzten 12 Monate:</p>
        <p style='font-size:30px'>- <a href='https://www.gdeltproject.org' target='_blank' style='color:white;'>Gdelt Project</a></p>
        <p style='font-size:30px'>- News APIs: NewsCatcher, NewsAPI, MediaStack</p>

    """, unsafe_allow_html=True)

    # Divider
    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Section: Methodik
    # Section: Methodik
    st.header("Methodik")
    st.markdown("<hr style='border:2px solid #5b5b5c'>", unsafe_allow_html=True)  # Section divider

    # Subsection: Themen Klassifizierung
    st.subheader("Themen Klassifizierung")
    st.markdown("""
        <p style='font-size:30px'>Modell: <a href='https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2' target='_blank' style='color:white;'>paraphrase-MiniLM-L6-v2</a></p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Politische Einordnung
    st.subheader("Politische Einordnung")
    st.markdown("""
        <p style='font-size:30px'>Medien wurden basierend auf 4 Quellen auf dem politischen Spektrum verordnet:</p>
        <p style='font-size:30px'>- <a href='https://interaktiv.tagesspiegel.de/lab/die-lieblingsmedien-der-parteien/' target='_blank' style='color:white;'>Tagesspiegel</a></p>
        <p style='font-size:30px'>- <a href='https://uebermedien.de/93000/wieso-haben-zeitungen-eine-politische-ausrichtung/' target='_blank' style='color:white;'>Übermedien</a></p>
        <p style='font-size:30px'>- <a href='https://www.pewresearch.org/global/fact-sheet/datenblatt-nachrichtenmedien-und-politische-haltungen-in-deutschland/' target='_blank' style='color:white;'>Pew Research</a></p>
        <p style='font-size:30px'>- <a href='https://www.polkom.ifp.uni-mainz.de/files/2024/01/pm_perspektivenvielfalt.pdf' target='_blank' style='color:white;'>Mainz Studie</a></p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Blinderfleck-Analyse
    st.subheader("Blinderfleck-Analyse")
    st.markdown("""
        <p style='font-size:30px'>Definiert als politische Ausrichtungen mit ≤10% Anteil an der Berichterstattung zum ausgewählten Thema.</p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Stimmungs-Analyse
    st.subheader("Stimmungs-Analyse")
    st.markdown("""
        <p style='font-size:30px'>- <a href='https://huggingface.co/ssary/XLM-RoBERTa-German-sentiment' target='_blank' style='color:white;'>XLM-RoBERTa-German-sentiment</a></p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Verwandte Themen
    st.subheader("Verwandte Themen")
    st.markdown("""
        <p style='font-size:30px'>Zeigt die 5 Themen mit den meisten (mind. 2) gemeinsamen Entitäten.</p>
    """, unsafe_allow_html=True)

    # Final Divider
    st.markdown("<hr style='border:2px solid #333'>", unsafe_allow_html=True)


    # Section: Datenbank
    st.header("Datenbank")
    st.markdown("<hr style='border:2px solid #5b5b5c'>", unsafe_allow_html=True)  # Section divider
    st.markdown(f"""
        <p style='font-size:30px; color:#f8f8fa'>
        Unsere Datenbank umfasst <b>{df.shape[0]:,} Artikel</b> aus <b>{df['source'].nunique()} Medien</b>, die <b>{df['topic'].nunique()} Themen</b> abdecken.
        </p>
    """, unsafe_allow_html=True)

    # Visualization suggestions
    st.markdown("""
        <p style='font-size:30px; color:#f8f8fa'>Hier sind einige Eckdaten visualisiert:</p>
    """, unsafe_allow_html=True)

    # Adjust plot fonts to size 30
    def create_colored_bar_chart(x, y, title, xaxis_title, yaxis_title):
        fig = go.Figure(data=[go.Bar(
            x=x,
            y=y,
            marker=dict(color=y, colorscale='dense'))])
        fig.update_layout(
            title=dict(text=title, font=dict(size=30)),
            xaxis=dict(title=xaxis_title, titlefont=dict(size=30), tickfont=dict(size=30)),
            yaxis=dict(title=yaxis_title, titlefont=dict(size=30), tickfont=dict(size=30)),
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"))
        return fig

    # Plot 1: Number of articles per political leaning
    st.markdown("<h3 style='font-size:30px; color:#5b5b5c'>Artikel nach politischer Ausrichtung</h3>", unsafe_allow_html=True)
    pol_leaning_counts = df['pol_leaning'].value_counts()
    fig1 = create_colored_bar_chart(
        x=pol_leaning_counts.index,
        y=pol_leaning_counts.values,
        title="",
        xaxis_title="",
        yaxis_title="Anzahl der Artikel")
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Number of articles per news outlet
    st.markdown("<h3 style='font-size:30px; color:#5b5b5c'>Artikel nach Nachrichtenquelle</h3>", unsafe_allow_html=True)
    source_counts = df['source'].value_counts()
    fig2 = create_colored_bar_chart(
        x=source_counts.index,
        y=source_counts.values,
        title="",
        xaxis_title="",
        yaxis_title="Anzahl der Artikel")
    st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Number of articles per topic
    st.markdown("<h3 style='font-size:30px; color:#5b5b5c'>Artikel nach Thema</h3>", unsafe_allow_html=True)
    topic_counts = df['topic'].value_counts()
    fig3 = create_colored_bar_chart(
        x=topic_counts.index,
        y=topic_counts.values,
        title="",
        xaxis_title="",
        yaxis_title="Anzahl der Artikel")
    st.plotly_chart(fig3, use_container_width=True)
