import numpy as np
import pickle as pkl
from rouge_score import rouge_scorer
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from metric import *
from plot import *
import math 

USE_Roberta = False
USE_EXACT_MATCH = True
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
if USE_Roberta:
    SenSimModel = SentenceTransformer('../data/weights/nli-roberta-large')

def printInfo(resultDict):
    print(len(resultDict))
    for item in resultDict:
        for key in item.keys():
            print(key)
        exit()

def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt

def getAccuracy(Label, Score, thresh):
    count = 0
    for ind, item in enumerate(Score):
        if item>=thresh and Label[ind]==1:
            count+=1
        if item<thresh and Label[ind]==0:
            count+=1
    return count/len(Score)

def getAcc(resultDict, file_name):
    correctCount = 0
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if rougeScore>0.5:
            correctCount += 1
    print("Acc:", 1.0*correctCount/len(resultDict))

def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]

def average_time(list):
    return sum(list) / len(list)  

def getAUROC(resultDict, file_name):
    Label = []
    Score = []
    Perplexity = []
    LexicalSimilarity = []
    Entropy = []
    AttentionScore = []
    CosineScore = []
    perplexity_time = []
    entropy_time = []
    attention_time = []
    cosine_time = []
    lexical_time = []

    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        Perplexity.append(-item["perplexity"])
        Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        AttentionScore.append(item["attention_score"]) # lexical similarity와 attention score는 높을수록 1, 낮을수록 0이므로 그대로 append, 나머지는 -를 붙여서 append
        CosineScore.append(item['cosine_score'])
        perplexity_time.append(item['perplexity_time'])
        entropy_time.append(item['entropy_time'])
        attention_time.append(item['attention_time'])
        cosine_time.append(item['cosine_time'])
        lexical_time.append(item['lexical_time'])
    # inf_indices = [i for i, x in enumerate(AttentionScore) if math.isinf(x)]
    # nan_indices = [i for i, x in enumerate(AttentionScore) if isinstance(x, float) and math.isnan(x)]
    # print(f"inf: {inf_indices}")
    # print(f"nan: {nan_indices}")

        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if rougeScore>0.5:
            Label.append(1) # non-hallucinate (correct)
        else:
            Label.append(0) # hallucinate (incorrect)
        Score.append(rougeScore)

    # infinity values occurs in AttentionScore
    AttentionScore[1193] = 1e10 # llama-7b SQuAD
    AttentionScore[1192] = 1e10 # llama-7b SQuAD
    # AttentionScore[1800] = 1e10 # llama-13b SQuAD
    # AttentionScore[3058] = 1e10 # llama-7b TriviaQA
    # AttentionScore[3055] = 1e10 # llama13b triviaqa

######### AUROC ###########
    fpr, tpr, thresholds = roc_curve(Label, Perplexity)
    AUROC = auc(fpr, tpr)
    # thresh_Perplexity = thresholds[np.argmax(tpr - fpr)]
    thresh_Perplexity = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Perplexity:", AUROC)
    # print("thresh_Perplexity:", thresh_Perplexity)
    VisAUROC(tpr, fpr, AUROC, "Perplexity")

    fpr, tpr, thresholds = roc_curve(Label, Entropy)
    AUROC = auc(fpr, tpr)
    # thresh_Entropy = thresholds[np.argmax(tpr - fpr)]
    thresh_Entropy = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Entropy:", AUROC)
    # print("thresh_Entropy:", thresh_Entropy)
    VisAUROC(tpr, fpr, AUROC, "NormalizedEntropy")

    fpr, tpr, thresholds = roc_curve(Label, LexicalSimilarity)
    AUROC = auc(fpr, tpr)
    # thresh_LexicalSim = thresholds[np.argmax(tpr - fpr)]
    thresh_LexicalSim = get_threshold(thresholds, tpr, fpr)
    print(f"lexical threshold: {thresh_LexicalSim}")
    print("AUROC-LexicalSim:", AUROC)
    # print("thresh_LexicalSim:", thresh_LexicalSim)
    VisAUROC(tpr, fpr, AUROC, "LexicalSim")

    fpr, tpr, thresholds = roc_curve(Label, AttentionScore)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScoreOutput = thresholds[np.argmax(tpr - fpr)]
    thresh_AttentionScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-AttentionScore:", AUROC)
    # print("thresh_EigenScoreOutput:", thresh_EigenScoreOutput)
    VisAUROC(tpr, fpr, AUROC, "AttentionScore", file_name.split("_")[1])

    fpr, tpr, thresholds = roc_curve(Label, CosineScore)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScoreOutput = thresholds[np.argmax(tpr - fpr)]
    thresh_CosineScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-CosineScore:", AUROC)
    # print("thresh_EigenScoreOutput:", thresh_EigenScoreOutput)
    VisAUROC(tpr, fpr, AUROC, "CosineScore", file_name.split("_")[1])
    
    rho_Perplexity = getPCC(Score, Perplexity)
    rho_Entropy = getPCC(Score, Entropy)
    rho_LexicalSimilarity = getPCC(Score, LexicalSimilarity)
    rho_AttentionScore = getPCC(Score, AttentionScore)
    rho_CosineScore = getPCC(Score, CosineScore)
    print("rho_Perplexity:", rho_Perplexity)
    print("rho_Entropy:", rho_Entropy)
    print("rho_LexicalSimilarity:", rho_LexicalSimilarity)
    print("rho_AttentionScore:", rho_AttentionScore)
    print("rho_CosineScore:", rho_CosineScore)
    
    acc = getAccuracy(Label, Perplexity, thresh_Perplexity)
    print("SQuAD Perplexity Accuracy:", acc)
    acc = getAccuracy(Label, Entropy, thresh_Entropy)
    print("SQuAD Entropy Accuracy:", acc)
    acc = getAccuracy(Label, LexicalSimilarity, thresh_LexicalSim)
    print("SQuAD LexicalSimilarity Accuracy:", acc)
    print("SQuAD EigenIndicator Accuracy:", acc)
    acc = getAccuracy(Label, AttentionScore, thresh_AttentionScore)
    print("SQuAD AttentionScore Accuracy:", acc)
    acc = getAccuracy(Label, CosineScore, thresh_CosineScore)
    print("SQuAD CosineScore Accuracy:", acc)

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

if __name__ == "__main__":
    file_name = "/mnt/aix7101/minsuh-output/llama-7b-hf_SQuAD_SDP&CS/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/llama-7b-hf_triviaqa_SDP&CS/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/llama-13b-hf_SQuAD_SDP&CS/0.pkl"
    # file_name = "/mnt/aix7101/minsuh-output/llama-13b-hf_triviaqa_SDP&CS/0.pkl"

    f = open(file_name, "rb")
    resultDict = pkl.load(f)
    # printInfo(resultDict)
    getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)

