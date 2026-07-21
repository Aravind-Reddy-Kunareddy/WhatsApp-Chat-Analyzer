from transformers import pipeline


emotion_classifier = pipeline(
    "text-classification",
    model="tabularisai/multilingual-emotion-classification"
)


def detect_emotion(message):

    result = emotion_classifier(
        message,
        truncation=True,
        max_length=512
    )

    return result

if __name__ == "__main__":

    test_messages = [
        "Bro today is very good",
        "Bhai kya bakwas hai",
        "Nuvvu enduku ila chestunnav ra",
        "😂😂😂",
        "I am very happy today",
        "Mujhe bahut gussa aa raha hai"
    ]

    for message in test_messages:

        result = detect_emotion(message)

        print("\nMessage:", message)
        print("Emotion:", result)