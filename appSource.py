from transformers import pipeline

max_new_tokens = 100

sentiment_analyzer = pipeline("sentiment-analysis")
text_generator = pipeline("text-generation", model="gpt2")


emotion_scores = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

def update_emotion_scores(input_text):
    
    global emotion_scores
    sentiment = sentiment_analyzer(input_text, truncation=True)
    sentiment_label = sentiment[0]['label']
    sentiment_score = sentiment[0]['score']

    # 스코어 누적
    emotion_scores[sentiment_label] += sentiment_score
    return sentiment_label, sentiment_score

def generate_response_and_prediction(emotion_scores, previous_statement):
    
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    
    if dominant_emotion == "POSITIVE":
        prompt_user = f"Your friend is happy based on their previous statement: '{previous_statement}'. Say something cheerful:"
    elif dominant_emotion == "NEGATIVE":
        prompt_user = f"Your friend seems upset based on their previous statement: '{previous_statement}'. Say something supportive:"
    else: 
        prompt_user = f"Your friend is neutral based on their previous statement: '{previous_statement}'. Say something engaging:"

    user_message = text_generator(
        prompt_user, 
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1, 
        truncation=True
    )[0]['generated_text'].strip()

    
    prompt_response = f"If someone said '{user_message}', the likely response would be:"
    predicted_response = text_generator(
        prompt_response, 
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1, 
        truncation=True
    )[0]['generated_text'].strip()

    return user_message, predicted_response

def process_input_text(input_text):
    """입력된 상대방의 말을 처리하고 추천 답변과 예상 반응 반환"""
    sentiment_label, sentiment_score = update_emotion_scores(input_text)
    user_message, predicted_response = generate_response_and_prediction(emotion_scores, input_text)
    
    return user_message, predicted_response, sentiment_label, sentiment_score


print("Enter the other person's text (type 'exit' to quit):")


while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break

    user_message, predicted_response, sentiment_label, sentiment_score = process_input_text(user_input)
    print(f"\n===================================\n[Recommended Reply]: {user_message}")
    print(f"\n\n[Expected Reaction]: {predicted_response}")
    print(f"\n\n[Emotion Score]: {sentiment_label} ({sentiment_score:.2f})\n")
