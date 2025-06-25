from keybert import KeyBERT 
from sentence_transformers import SentenceTransformer

def get_key(messages):
    texts=[]

    for message in messages:
        texts.append(message["question"])

    candidates = ["재정","주식","투자","저축","예금",
                "대출","채권","자산","금리","경제","환율",
                "이율","보험","월세","적금"]

    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    kw_model = KeyBERT(model)
    keyword=kw_model.extract_keywords(
            texts,
            keyphrase_ngram_range=(1,1),
            stop_words=None,
            use_mmr=True,
            diversity=0,
            candidates=candidates,
            top_n=10
        )
    key = [kw for sublist in keyword for (kw, _) in sublist]
    return key