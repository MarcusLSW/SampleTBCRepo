import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

token = 'xoxp-741285264438-738634010148-813678252069-abf2d74fbf3f86d71362ff11201de2ae'

# get the channels that are readable for the slack group
params = {'token': token}
response = requests.get("https://slack.com/api/channels.list", params=params)
channels = response.json()['channels']
channel_to_read = None
for channel in channels:
    if channel['name'] == 'capstone':
        channel_to_read = channel

print(channel_to_read)
# scanner through the channel
params_history = {'token': token, 'channel': channel_to_read['id']}
response_history = requests.get("https://slack.com/api/channels.history", params=params_history)

has_next = True
messages_list = []
while has_next:
    response_history = response_history.json()
    has_next = response_history['has_more']
    messages = response_history['messages']
    latest = messages[len(messages) - 1]['ts']
    params_history['latest'] = latest
    response_history = requests.get("https://slack.com/api/channels.history", params=params_history)

    messages_list.extend(messages)
    time.sleep(1)

text_list = []
for message in messages_list:
    text_list.append(message['text'])

# stop words
stop_words = set(stopwords.words('english'))
vectorizer_tfidf = TfidfVectorizer(stop_words=stop_words, lowercase=True)
bow_tfidf = vectorizer_tfidf.fit_transform(text_list).toarray()

features = vectorizer_tfidf.get_feature_names()
stop_words = stop_words.union(features[0:features.index('able')])

vectorizer_tfidf = TfidfVectorizer(stop_words=stop_words, lowercase=True)
bow_tfidf = vectorizer_tfidf.fit_transform(text_list).toarray()
# summed and flattened matrix
sums = [sum(x) for x in zip(*bow_tfidf)]
features = vectorizer_tfidf.get_feature_names()

# intialise data of lists.
data = {'features': features, 'sum': sums}
# Create DataFrame
df = pd.DataFrame(data)
# Print the output.
df.sort_values(by=['sum'], ascending=False)

wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='black').generate(str(df.values))
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
