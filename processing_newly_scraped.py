import logging  # For logging information and errors
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For handling NaN values and numerical operations
import re  # For regular expressions to clean text
from collections import Counter  # For counting entity occurrences
import nltk  # For natural language processing tasks
from nltk.corpus import stopwords  # For stopword removal
from gensim.utils import simple_preprocess  # For text preprocessing
from transformers import AutoTokenizer, pipeline  # For sentiment analysis pipeline

# Configure logging to include timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S')

nltk.download('stopwords')
logging.info("-----------> all libs imported")

#--------------------------------------
# cleaning freshly scraped article data
#--------------------------------------
#df = combined_df
df = pd.read_csv('/Users/reppmazc/Downloads/df_combined_scraped_new_content_entities.csv')

# renaming
#---------
df = df.rename(columns={'publishedAt': 'datetime'})

# format date time
#-----------------
def format_datetime_string(dt_string):
    # remove timezone offset
    dt_string = dt_string.split('+')[0]
    # remove 'T'
    dt_string = dt_string.replace('T', '').replace('-', '').replace(':', '').replace(' ', '')
    return dt_string

# apply func
df['datetime'] = df['datetime'].astype(str).apply(format_datetime_string)

# remove potential dublicates
#----------------------------
df = df.drop_duplicates(subset='url', keep='last')

# replace replace weird signs with nan
#-------------------------------------
freaky_pattern = r'Ã¼Ã¤Ã¶ÃŸâ¬Ã'
df['content'] = df['content'].apply(lambda x: np.nan if pd.notna(x) and bool(re.search(freaky_pattern, x)) else x)

# check if text is german
#------------------------
# common German words
common_german_words = [
    "und", "der", "das", "ist", "zu", "mit", "von", "auf", "für", "den", "im", "ein", "nicht",
    "eine", "als", "auch", "aber", "wie", "es", "am", "aus", "bei", "dass", "oder", "so", "wenn",
    "werden", "wir", "hat", "sich", "dem", "des", "noch", "nur", "kann", "um", "ja", "mehr"]

threshold = 5

def is_german(text):
    # count German words
    word_count = sum(word in text.lower() for word in common_german_words)
    # return NaN if text doesnt meet german word threshold
    return text if word_count >= threshold else np.nan

# apply func
df['content'] = df['content'].apply(lambda x: is_german(x) if isinstance(x, str) else np.nan)

# remove random strings, signs and symbols
#-----------------------------------------
non_article_snippets = [
    'Hauptnavigation: Nutzen Sie die Tabulatortaste, um durch die Menüpunkte zu navigieren. Öffnen Sie Untermenüs mit der Leertaste. Schließen Sie Untermenüs mit der Escape-Taste. Hauptnavigation: Nutzen Sie die Tabulatortaste, um durch die Menüpunkte zu navigieren. Öffnen Sie Untermenüs mit der Leertaste.',
    'Lesen Sie mehr zum Thema In anspruchsvollen Berufsfeldern im Stellenmarkt der SZ. Sie möchten die digitalen Produkte der SZ mit uns weiterentwickeln? Bewerben Sie sich jetzt!Jobs bei der SZ Digitale Medien Gutscheine:',
    'öffnet in neuem Tab oder Fenster',
    'Danke, dass Sie ZEIT ONLINE nutzen.',
    'Melden Sie sich jetzt mit Ihrem bestehenden Account an oder testen Sie unser digitales Abo mit Zugang zu allen Artikeln. Erscheinungsbild Die Zusammenfassung für diesen Artikel kann leider momentan nicht angezeigt werden.',
    'ZEIT ONLINE hat diese Meldung redaktionell nicht bearbeitet. Sie wurde automatisch von der Deutschen Presse-Agentur (dpa) übernommen.',
    'Kommentar | ',
    'Berlin (dpa/bb). ',
    'Deutschen Presse-Agentur',
    'dpa',
    '+++ ',
    'SZ-Redaktion',
    'WELT TV'
    '++',
    '© dpa-infocom, ',
    'Drucken Teilen',
    'Nicht verpassen: Alles rund ums Thema Job & Beruf finden Sie im Karriere-Newsletter unseres Partners Merkur.de.',
    'Erstellt durch: ',
    '0 Weniger als eine Minute',
    '► ',
    '▶︎'
    '© ',
    '© dpa/',
    '© REUTERS/',
    '© Getty Images',
    '© IMAGO/',
    '© imago images/',
    '© imago/',
    'Lesen Sie auch',
    '© Berliner Feuerwehr ',
    '© privat ',
    'Kopiere den aktuellen Link ',
    'DIE ZEIT: ',
    'Melden Sie sich jetzt mit Ihrem bestehenden Account an oder testen Sie unser digitales Abo mit Zugang zu allen Artikeln. Abo testen',
    '"Licht Aus" So ging es Jochen Schropp nach der Sendung Im Interview verrät Jochen Schropp, was "Licht Aus" in ihm verändert hat und was für ihn zu den schönsten Erlebnissen aus der Show zählt.',
    '"Mehr als zwei Etagen schaffe ich kaum zu Fuß" Baerbock schildert im stern Folgen ihrer Covid-Erkrankung – viele Genesungswünsche für Außenministerin',
    '(SZ) ',
    '06. November 2024: Til Schweiger postet Schwarz-Weiß-Foto von sich – Fans denken er sei tot Mit seinem neuen Posting auf Instagram wollte Til Schweiger eigentlich nur ein Update geben. Doch das Foto, welches er für seinen Instagram-Beitrag gewählt hat, sorgt unter seinen Fans für Verwunderung. Denn das Schwarz-Weiß-Bild erinnert an eine Todesanzeige. "Was ist passiert? Es soll dumm aussehen, aber solche Fotos postet man gewöhnlich nur als ein Todeszeichen im sozialen Netzwerk", schreibt ein Fan. "Ist er gestorben?", fragt ein anderer. Andere freuen sich, überhaupt mal wieder etwas von Schweiger zu erfahren. Screenshot Instagram Til Schweiger Mehr',
    '167 Millionen Aufrufe: Trailer des neuen "Joker"-Films weckt Rieseninteresse Schon ein halbes Jahr vor seinem Kinostart weckt die Fortsetzung des Thrillers "Joker" riesiges Interesse. Der Trailer von "Joker: Folie A Deux" mit Popstar Lady Gaga in einer Hauptrolle wurde innerhalb der ersten 24 Stunden nach seiner Veröffentlichung sensationelle 167 Millionen Mal angesehen, wie das Branchenmagazin "Variety" am Donnerstag (Ortszeit) berichtete. Auf der Videoplattform Youtube landete der Trailer demnach sofort an der Spitze der Trendvideos.',
    '1999 in Hamburg verschwunden Polizei sucht mysteriösen Anrufer – neue Spur im Vermisstenfall Hilal Ercan Sie ist das einzige langzeitvermisste Kind in Hamburg: 1999 verschwand die damals zehnjährige Hilal Ercan spurlos. Jetzt verfolgt die Polizei eine neue Spur.',
    'AFP/',
    'An dieser Stelle ist ein externer Inhalt eingebunden Zum Anschauen benötigen wir Ihre Zustimmung Bitte aktivieren Sie JavaScript damit Sie diesen Inhalt anzeigen können. Weiter ',
    'Andreas Klaer ',
    'Auf Diddys Jacht Neue Missbrauchsvorwürfe gegen Sänger Chris Brown In einer Doku sollen Frauen zu Wort kommen, die Chris Brown schwere Vorwürfe machen. Er wird unter anderem der Vergewaltigung beschuldigt – nicht zum ersten Mal.',
    '(dpa/tmn). ',
    'Bornschein Podcast : Bornschein trifft Henrik Falk Der Strategie- und Digitalexperte Christoph Bornschein sucht mit eingeladenen Fachleuten nach Ideen der Zukunft. Es geht oft ums Digitale, aber weit darüber hinaus. Diese Woche ist der Vorstandsvorsitzende der Berliner Verkehrsbetriebe zu Gast. Es geht um die digitale Verkehrswende. Und darum, was das überhaupt ist.',
    'Das stern-Team vor Ort informiert Sie immer samstags im kostenlosen Newsletter "Inside America" über die wichtigsten Entwicklungen und gibt Einblicke, wie Amerikanerinnen und Amerikaner wirklich auf ihr Land schauen. Hier geht es zur Registrierung. Nach Eingabe Ihrer E-Mail-Adresse erhalten Sie eine E-Mail zur Bestätigung Ihrer Anmeldung.',
    'Der F.A.Z. Podcast für Deutschland ist der tägliche Podcast der F.A.Z. zu den relevantesten Themen des Tages. Der Podcast erscheint immer um 17 Uhr, von Montag bis Freitag. Alle Folgen finden Sie hier . Sie können den Podcast ganz einfach bei Apple Podcasts, Spotify oder Deezer abonnieren und verpassen so keine neue Folge. Natürlich sind wir auch in anderen Podcast-Apps verfügbar, suchen Sie dort einfach nach „F.A.Z. Podcast für Deutschland“. Ebenfalls finden Sie uns in der FAZ.NET-App.',
    'Dieser Artikel ist Teil von ZEIT am Wochenende, ',
    'Direkt aus dem dpa-Newskanal: Dieser Text wurde automatisch von der Deutschen Presse-Agentur (dpa) übernommen und von der SZ-Redaktion nicht bearbeitet. ',
    'dpa/- ',
    'dpa/ ',
    'dpa/',
    '+ ',
    'F.A.Z. exklusiv : ',
    'Gestaltung: Tagesspiegel; Fotos: ',
    'imago images / ',
    'imago images/',
    'IMAGO/',
    'Imago/',
    'imago/',
    'Inhalt Auf einer Seite lesen Inhalt Seite 1 ',
    'Interview | ',
    'Jetzt im Livestream – ',
    'Kennen Sie schon unsere PLUS-Inhalte ? Jetzt Morgenpost testen Sie haben vermutlich einen Ad-Blocker aktiviert. Aus diesem Grund können die Funktionen des Podcast-Players eingeschränkt sein. Bitte deaktivieren Sie den Ad-Blocker, um den Podcast hören zu können.',
    'Jetzt weiterlesen Dies ist kein Abo. Ihre Registrierung ist komplett kostenlos, ohne versteckte Kosten. Gleich geschafft! Bitte bestätigen Sie Ihren Account über den Bestätigungslink in der gesendeten E-Mail',
    'Kopiere den aktuellen',
    'liveblog ',
    'Netflix entfernt interaktive Titel Was User jetzt noch sehen müssen Netflix nimmt die meisten interaktiven Filme und Serien aus dem Katalog. Was Abonnenten jetzt noch sehen müssen.',
    'News von ZDFheute on Instagram: ',
    'Ottmar Winter PNN/Ottmar Winter PNN ',
    'picture alliance / ',
    'picture alliance/dpa ',
    '(dpa/bb).',
    'rbb|24-Adventskalender | ',
    'REUTERS/',
    'Reuters/',
    'Teilen Verschenken Merken Drucken ',
    'Von: '
    'Schließen Artikelzusammenfassung Dies ist ein experimentelles Tool. Die Resultate können unvollständig, veraltet oder sogar falsch sein. ',
    'Sehen Sie im Video: ',
    'Sie können den Podcast ganz einfach bei Apple Podcasts, Spotify oder Deezer abonnieren und verpassen so keine neue Folge. Natürlich sind wir auch in anderen Podcast-Apps verfügbar, suchen Sie dort einfach nach „F.A.Z. Podcast Finanzen & Immobilien“. Ebenfalls finden Sie uns in der FAZ.NET-App. Alle unsere Podcast-Angebote finden Sie hier. Haben Sie Fragen oder Anmerkungen zum Podcast? Dann melden Sie sich gerne bei podcast@faz.de.',
    'Sie sind nun eingeloggt. Sollten Sie dennoch nicht auf gesperrte Artikel zugreifen können, löschen Sie bitte die im Browser gespeicherten Cookies und loggen Sie sich dann erneut ein. Zum Artikel: ']

phrases_to_trim_after = [
    'Die WELT als ePaper: Die vollständige Ausgabe steht Ihnen bereits am Vorabend zur Verfügung – so sind Sie immer hochaktuell informiert. Weitere Informationen http://epaper.welt.de Der Kurz-Link dieses Artikels lautet:',
    'Hier können Sie interessante Artikel speichern',
    'Wiesbaden/Schwalmstadt. Nach den tödlichen Polizeischüssen in Schwalmstadt hat Innenminister Roman Poseck (CDU) den',
    'Sie wollen wissen, wie die Sterne in Liebesdingen, im Beruf und in Sachen Gesundheit für Sie stehen? ',
    'Prime Deal Days 2024: Das waren die besten Angebote Kopiere den aktuellen Link Amazon',
    'LED-Taschenlampe für 18 statt 37 Euro: Die Top-Deals am Donnerstag Amazon',
    'Kaufberatung Audi A5 - diese Schöheit ist kein Biest Wieso Audi so lange keinen echten Konkurrente',
    'Einhell Rasentrimmer heute 41 % günstiger: ',
    ' <blockquote class="instagram-media" data-instgrm-captioned data-',
    'Haben Sie den Eurojackpot geknackt?',
    'Tag für Tag gibt es Ereignisse, Anekdoten, Geburts- oder Sterbetage, an die erinnert werden soll. ',
    'Beim Lotto 6aus49 steht die Gewinnchance bei 1:140 Millionen Jeder träumt von einem 6er im Lotto ... ',
    'Berlin. Das Cashplus von Unitplus bietet aktuell 3,23 Prozent Zinsen und damit mehr als manches Tagesgeld. ',
    'Berlin. Für einen Auto-, Wohn- oder Ratenkredit der ING gelten neue Konditionen. ',
    'Berlin. Zinsstarke Angebote für ein Festgeld oder Tagesgeld werden rar. ',
    'Die Gewinnchancen beim Lotto 6aus49 stehen bei ',
    'Aktuelle Nachrichten und Hintergründe aus Politik, Wirtschaft und Sport aus Berlin, Deutschland und der Welt.']

# remove specified snippets from within the content
def remove_snippets(text):
    if pd.notna(text):
        for snippet in non_article_snippets:
            text = text.replace(snippet, '')
    return text

df['content'] = df['content'].apply(remove_snippets)

# trim content after any of the specified phrases
def trim_content(text):
    if pd.notna(text):
        for phrase in phrases_to_trim_after:
            if phrase in text:
                return text.split(phrase)[0]
    return text

df['content'] = df['content'].apply(trim_content)

# remove signs, symbols, etc.
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        text = re.sub(r"http\S+", "", text)  # URLs
        text = re.sub(r"©\s?\d+", "", text)  # copyright symbols
        text = re.sub(r'\n+', ' ', text)  # replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces
        return text
    return text

# Apply the cleaning function to each article
df['content'] = df['content'].apply(clean_text)

# deal with nans and empty cells
#-------------------------------
# nan in cells with only whitespace
df['content'] = df['content'].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)

# remove nans
df = df.dropna()

# create new source column from url
#-----------------------------------
df['source'] = df['url'].str.extract(r'https?://(?:www\.)?([\w-]+)\.')

# removed any non-approved sources
#-----------------------------------
allowed_sources = ["spiegel",
                   "taz",
                   "jungewelt",
                   "freitag",
                   "zeit",
                   "tagesspiegel",
                   "dw",
                   "tagesschau",
                   "stern",
                   "focus",
                   "welt",
                   "jungefreiheit",
                   "sueddeutsche",
                   "faz",
                   "morgenpost",
                   "bild",
                   "rbb24",
                   "express"]

df = df[df['source'].isin(allowed_sources)]

logging.info('-----------> cleaning done')
print(df['source'].value_counts())

#-----------------------------------------
# assign political leaning based on source
#-----------------------------------------
# source to political leaning mapping
source_to_label = {"spiegel": "mitte_links",
                   "taz": "links",
                   "jungewelt": "links",
                   "freitag": "links",
                   "zeit": "mitte_links",
                   "tagesspiegel": "mitte_links",
                   "dw": "mitte",
                   "tagesschau": "mitte",
                   "stern": "mitte_rechts",
                   "focus": "mitte_rechts",
                   "welt": "mitte_rechts",
                   "jungefreiheit": "rechts",
                   "sueddeutsche": "mitte_links",
                   "faz": "mitte_rechts",
                   "morgenpost": "mitte_rechts",
                   "bild": "rechts",
                   "rbb24": "mitte",
                   "express": "rechts"}

df['pol_leaning'] = df['source'].map(source_to_label)

logging.info('-----------> pol leaning done')

#--------------------------------------------------------
# preproc for topic classification and sentiment analysis
#--------------------------------------------------------
custom_stopwords = ['']
stop_words = set(stopwords.words('german'))

def preprocess_text(text):
    # Remove punctuation and numbers, lowercase, and remove stopwords
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    tokens = simple_preprocess(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

df['processed_content'] = df['content'].apply(preprocess_text)
logging.info('-----------> preproc for classification and sentiment done')

#-------------------
# sentiment analysis
#-------------------
# load tokenizer and sentiment analysis pipeline
model_name = "ssary/XLM-RoBERTa-German-sentiment"
max_token_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis",
                              model=model_name,
                              tokenizer=tokenizer,
                              max_length=max_token_length,
                              truncation=True,
                              device=0)

# truncate text function
def truncate_text(text):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_token_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# apply truncation to  processed content
df['processed_content_truncated'] = df['processed_content'].apply(truncate_text)

# apply the sentiment pipeline to each row
batch_size = 10
sentiment_results = []

for start in range(0, len(df), batch_size):
    batch_texts = df['processed_content_truncated'][start:start+batch_size].tolist()
    batch_sentiments = sentiment_pipeline(batch_texts)
    sentiment_results.extend([result['label'] for result in batch_sentiments])

df['sentiment'] = sentiment_results

logging.info("-----------> sentiment analysis done")

#---------------
# rename columns
#---------------
df = df.rename(columns={'topic': 'topic_nr',
                        'assigned_label': 'topic',
                        'entity_counts': 'entities'})

#---------------------------------------------------
# generating related topics column based on entities
#---------------------------------------------------
grouped = df.groupby('topic')

# get top 10 entities for each topic and store them in a dict
top_entities_by_topic = {}

for topic, group in grouped:
    # combine all entities across articles for this topic
    all_entities = Counter()
    for entities in group['entities']:
        entity_dict = eval(entities) if isinstance(entities, str) else entities
        all_entities.update(entity_dict)
    
    # get top 10 entities for this topic
    top_entities = [entity for entity, count in all_entities.most_common(10)]
    top_entities_by_topic[topic] = top_entities

# ceate a related_topics column with overlap count
def find_related_topics(topic):
    # find topics with overlapping entities in top 10 entities
    related = [
        (other_topic, len(set(top_entities_by_topic[topic]).intersection(entities)))
        for other_topic, entities in top_entities_by_topic.items() 
        if other_topic != topic and len(set(top_entities_by_topic[topic]).intersection(entities)) > 0]
    return related

# apply func to each row
df['related_topics'] = df['topic'].apply(find_related_topics)

logging.info('-----------> related topics done')
#---------------------------
# merge with historical data
#---------------------------
df.to_csv('test.csv')
logging.info('-----------> doneeeeee')