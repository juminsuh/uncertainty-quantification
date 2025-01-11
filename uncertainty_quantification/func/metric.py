import numpy as np
from numpy.linalg import norm
import torch

from rouge_score import rouge_scorer
from sentence_transformers import util
from itertools import combinations

rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def getRouge(rouge, generations, answers):
    # results = rouge.compute(predictions=[generations], references=[answers], use_aggregator=False)
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure  #fmeasure/recall/precision
    return RoughL

def get_perplexity_score(scores):
    perplexity = 0.0
    for logits in scores:
        conf = torch.max(logits.softmax(1)).cpu().item()
        perplexity += np.log(conf)
    perplexity = -1.0 * perplexity/len(scores)
    return perplexity

### batch_scores ([[logits]], [[logits]], [[logits]])
### num_tokens : list 
def get_entropy_score(batch_scores, num_tokens):  
    Conf = []
    for logits in batch_scores:
        conf, index = torch.max(logits.softmax(1), dim=1)
        Conf.append(conf.cpu().numpy())
    Conf = np.array(Conf)  
    Conf = Conf + 1e-6
    entropy = -1.0 * np.sum(np.log(Conf))/logits.shape[0]
    return entropy

def get_lenghthNormalized_entropy(batch_scores, num_tokens):  
    seq_entropy = np.zeros(len(num_tokens))  
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind]
    normalized_entropy = -1.0* normalized_entropy/len(num_tokens)
    return normalized_entropy

def getLexicalSim(generated_texts):
    LexicalSim = 0
    for i in range(len(generated_texts)):
        for j in range(len(generated_texts)):
            if j<=i:
                continue
            LexicalSim += getRouge(rougeEvaluator, generated_texts[i], generated_texts[j])
    LexicalSim = LexicalSim/(len(generated_texts)*(len(generated_texts)-1)/2)
    return LexicalSim

def compute_AttentionScore(hidden_states, num_tokens, args, iter):
    
    selected_layer = int(len(hidden_states[0])/2) # the number of layers = int(33/2) = 16
    if len(hidden_states) < 2:
        return None
    concatenated_matrix = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda:0") # (10, 4096)
    for idx in range(hidden_states[1][-1].shape[0]): # 10번
        concatenated_matrix[idx,:] = hidden_states[num_tokens[idx]-2][selected_layer][idx,0,:] # 마지막 토큰의 중간 레이어의 idx번째 sequence의 embedding_size
    concatenated_matrix = concatenated_matrix.unsqueeze(0) # (1, 10, 4096)

    concat_matrix = concatenated_matrix.to(args.device).half() # float16으로 변환 (계산의 효율성)
    num_gens = concat_matrix.size(1) # 10
    embedding = concat_matrix.size(2) # 4096
    sdp_map = torch.matmul(concat_matrix, concat_matrix.permute(0, 2, 1)) / torch.sqrt(torch.tensor(embedding, dtype=torch.float16)) # (1, 10, 10)

    upper_sdp = torch.triu(sdp_map, diagonal = 1)  # k=1은 주대각선을 제외하고 상삼각행렬 추출
    upper_sdp_values = upper_sdp[upper_sdp != 0] # 상삼각행렬만을 요소로 갖는 텐서
    
    output = upper_sdp_values.sum()
    denominator = (num_gens*(num_gens-1))//2 # 45
    output = output / denominator 
    
    return output.item() # 텐서의 값을 실수 값으로 return

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def compute_CosineSimilarity(hidden_states, num_tokens):
    selected_layer = int(len(hidden_states[0])/2) # the number of layers = int(33/2) = 16
    if len(hidden_states) < 2:
        return None
    concatenated_matrix = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda:0") # (10, 4096)
    for idx in range(hidden_states[1][-1].shape[0]): # 10번
        concatenated_matrix[idx,:] = hidden_states[num_tokens[idx]-2][selected_layer][idx,0,:] # 마지막 토큰의 중간 레이어의 idx번째 sequence의 embedding_size, (10, 4096)

    concatenated_matrix = concatenated_matrix.cpu().numpy().astype(float) # float64
    num_gens = concatenated_matrix.shape[0] # 10
    indices = list(range(num_gens))
    combinated = list(combinations(indices, 2)) # indices들을 2개씩 조합을 만듦
    cosine_similarities = []

    for i, j in combinated:
        similarity = cosine_similarity(concatenated_matrix[i], concatenated_matrix[j])
        cosine_similarities.append(similarity)
    
    output = sum(cosine_similarities)
    denominator = (num_gens*(num_gens-1))//2
    output = output/denominator

    return output




    







