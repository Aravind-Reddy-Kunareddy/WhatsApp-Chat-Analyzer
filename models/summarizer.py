import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def clean_messages(messages):

    cleaned_messages = []

    for message in messages:

        message = str(message).strip()

        # Remove media
        if "<Media omitted>" in message:
            continue

        # Remove Instagram and other links
        if "http" in message.lower():
            continue

        if "This message was deleted" in message:
            continue

        # Remove edited message tag
        message = message.replace(
            "<This message was edited>",
            ""
        ).strip()

        # Remove empty messages
        if len(message) == 0:
            continue

        # Remove very short messages
        if len(message) < 10:
            continue

        # Remove number-only messages
        if re.fullmatch(
            r'[\d\s.,-]+',
            message
        ):
            continue

        # Remove emoji-only messages
        if not re.search(
            r'[a-zA-Z]',
            message
        ):
            continue

        cleaned_messages.append(message)

    return cleaned_messages

def detect_topics(messages, n_topics=5):

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000
    )

    X = vectorizer.fit_transform(messages)

    kmeans = KMeans(
        n_clusters=n_topics,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    return clusters, vectorizer, kmeans


def get_topic_keywords(
    vectorizer,
    kmeans,
    n_words=5
):

    terms = vectorizer.get_feature_names_out()

    topic_keywords = {}

    for topic_id, center in enumerate(
        kmeans.cluster_centers_
    ):

        top_indices = center.argsort()[
            -n_words:
        ][::-1]

        keywords = [
            terms[index]
            for index in top_indices
        ]

        topic_keywords[topic_id] = keywords

    return topic_keywords


if __name__ == "__main__":

    import preprocessor

    with open(
        "Group_chat.txt",
        "r",
        encoding="utf-8"
    ) as file:

        data = file.read()

    df, chat_type = preprocessor.preprocess(data)

    cleaned_messages = clean_messages(
        df['messages']
    )

    clusters, vectorizer, kmeans = detect_topics(
        cleaned_messages,
        n_topics=5
    )

    topic_keywords = get_topic_keywords(
        vectorizer,
        kmeans
    )

    print("\nTOPIC GROUPS:\n")

    for topic in range(5):

        print("\n==============================")
        print(
            "TOPIC",
            topic,
            "→",
            ", ".join(
                topic_keywords[topic]
            )
        )
        print("==============================")

        topic_messages = [
            cleaned_messages[i]
            for i in range(
                len(cleaned_messages)
            )
            if clusters[i] == topic
        ]

        for message in topic_messages[:10]:

            print("-", message)