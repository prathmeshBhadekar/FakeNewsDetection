with open('tfidf_bias.pkl', 'rb') as f:
    tfidf_bias = pickle.load(f)

with open('passive_bias.pkl', 'rb') as f:
    passive_bias = pickle.load(f)

with open('svc_bias.pkl', 'rb') as f:
    svc_bias = pickle.load(f)


def callbias(bias_text, bias_tfidf, passive_bias):
    bias_transform = tfidf_bias.transform([bias_text])
    bias_output = passive_bias.predict(bias_transform)
    return bias_output
