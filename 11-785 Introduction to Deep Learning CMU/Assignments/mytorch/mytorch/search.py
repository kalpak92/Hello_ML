import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    forward_prob = 1.0
    forward_path = " "

    #print(y_probs.shape, y_probs)
    #print(SymbolSets)
    SymbolSets = ['-'] + SymbolSets
    for t in range(y_probs.shape[1]):
        #print("Current t: ", y_probs[:,t,:])
        i_t = np.argmax(y_probs[:,t,:],axis=0)[0]
        s_t = SymbolSets[i_t][0]
        #print("i_t: ", i_t, "s_t: ", s_t)
        forward_prob *= y_probs[i_t,t,:]
        if forward_path[-1] != s_t:# and s_t != '-':
            forward_path += s_t
    # return (forward_path, forward_prob)

    return (forward_path[1:].replace('-',''),forward_prob)



##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''

PathScore = {}
BlankPathScore = {}

def Prune(path_b, path_s, BlankPathScore, PathScore, beam_width):

    pruned_blank_pscore = {}
    Pruned_pscore = {}
    score_list = []

    for p in path_b:
        score_list.append(BlankPathScore[p])
    for p in path_s:
        score_list.append(PathScore[p])
    
    #sort and find cutoff
    if beam_width < len(score_list):
        cutoff = sorted(score_list, reverse=True)[beam_width - 1]
    else:
        cutoff =  sorted(score_list, reverse=True)[-1]
    
    pruned_path_b = []
    #print("Debug:", cutoff, beam_width)
    #print("Debug:", blank_pscore)
    for p in path_b:
        if BlankPathScore[p] >= cutoff:
            #print("Debug:", BlankPathScore[p] , p )
            pruned_path_b.append(p)
            pruned_blank_pscore[p] = BlankPathScore[p]
    
    pruned_path_s = []
    for p in path_s:
        if PathScore[p] >= cutoff:
            pruned_path_s.append(p)
            Pruned_pscore[p] = PathScore[p]
    
    return pruned_path_b, pruned_path_s, pruned_blank_pscore, Pruned_pscore
    
def InitializePaths(SymbolSets, y):
    #print(y.shape)

    initial_blank_pscore = {}
    initial_pscore = {}
    
    path = ""
    initial_blank_pscore[path] = y[0]
    initial_path_b = [path]
    
    initial_path_s = []
    for c_i, c in enumerate(SymbolSets):
        path = c
        initial_pscore[c] = y[c_i + 1]
        initial_path_s.append(path)
    
    return initial_path_b, initial_path_s, initial_blank_pscore, initial_pscore

def ExtendWithBlank(p_b, p_s, y):
    global PathScore
    global BlankPathScore
    updated_path_b = []
    updated_pscore = {}
    #work with terminal blanks
    for p in p_b:
        updated_path_b.append(p)
        updated_pscore[p] = BlankPathScore[p] * y[0]
    
    #work with terminal symbols 
    for p in p_s:
        if p in updated_path_b:
            updated_pscore[p] += PathScore[p] * y[0]
        else:
            updated_path_b.append(p)
            updated_pscore[p] = PathScore[p] * y[0]

    return updated_path_b, updated_pscore
    
def ExtendWithSymbol(p_b, p_s, SymbolSets, y):
    global PathScore
    global BlankPathScore
    updated_path_s = []
    updated_pscore = {}

    #work with terminating blanks
    for p in p_b:
        for c_i, c in enumerate(SymbolSets):
            new_path = p + c
            updated_path_s.append(new_path)
            updated_pscore[new_path] = BlankPathScore[p] * y[c_i + 1]

    #work with terminating symbols
    for p in p_s:
        for c_i, c in enumerate(SymbolSets):
            if c != p[-1]:
                new_path = p + c
            else:
                new_path = p
            if new_path in updated_path_s:
                updated_pscore[new_path] += PathScore[p] * y[c_i + 1]
            else:
                updated_path_s.append(new_path)
                updated_pscore[new_path] = PathScore[p] * y[c_i + 1]

    return updated_path_s, updated_pscore

def MergeIdenticalPaths(p_b, blank_pscore, p_s, pscore):
    MergedPaths = p_s
    FinalPathScore = pscore

    for p in p_b:
        if p in MergedPaths:
            FinalPathScore[p] += blank_pscore[p]
        else:
            MergedPaths.append(p)
            FinalPathScore[p] = blank_pscore[p]
    return MergedPaths, FinalPathScore

def BeamSearch(SymbolSets, y_probs, BeamWidth):

    
    global PathScore
    global BlankPathScore
    
    
    #initialize paths
    new_path_b , new_path_s, new_blank_pscore, new_pscore = InitializePaths(SymbolSets, y_probs[:,0,0])
    #print(new_path_b , new_path_s, new_blank_pscore, new_pscore)

    for t in range(1,y_probs.shape[1]):
        #print("Iter t:, ", t)
        #prune the collection down to BeamWidth
        path_b, path_s, BlankPathScore, PathScore = Prune(new_path_b , new_path_s, new_blank_pscore, new_pscore, BeamWidth)

        #extend paths by a blank
        new_path_b, new_blank_pscore = ExtendWithBlank(path_b, path_s, y_probs[:,t,0])

        #extend paths by a symbol
        new_path_s, new_pscore =  ExtendWithSymbol(path_b, path_s, SymbolSets, y_probs[:,t,0])


    #path_b, path_s, BlankPathScore, PathScore = Prune(new_path_b , new_path_s, new_blank_pscore, new_pscore, BeamWidth)
        
    MergedPath, mergedPathScores =  MergeIdenticalPaths(new_path_b, new_blank_pscore, new_path_s, new_pscore)
    bestPath = sorted(mergedPathScores.items(), key=lambda kv: kv[1])[-1][0]
   
    #sorted_x = sorted(mergedPathScores.items(), key=lambda kv: kv[1])
    #print(sorted_x[-1])
    # Follow the pseudocode from lecture to complete beam search :-)
    #bestPath = sorted_x[-1]
    #print(bestPath, mergedPathScores)
    return (bestPath, mergedPathScores)#{bestPath:mergedPathScores[bestPath]})
    #raise NotImplementedError




