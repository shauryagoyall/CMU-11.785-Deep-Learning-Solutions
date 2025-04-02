import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # (num_symbols+1, seq_length, batch_size)
        num_symbols_plus_blank, seq_length, batch_size = y_probs.shape

        # For now, we assume batch_size is 1.
        previous_symbol = None

        # Iterate over sequence length
        for t in range(seq_length):
            timestep_probs = y_probs[:, t, 0]

            best_symbol_index = np.argmax(timestep_probs)
            best_symbol_prob = timestep_probs[best_symbol_index]

            path_prob *= best_symbol_prob

            if best_symbol_index != blank and best_symbol_index != previous_symbol:
                decoded_path.append(self.symbol_set[best_symbol_index - 1])

            previous_symbol = best_symbol_index

        decoded_path_str = ''.join(decoded_path)
        return decoded_path_str, path_prob


def InitializePaths(SymbolSets, y):
    InitialBlankPathScore, InitialPathScore = {}, {}
    path = ''
    InitialBlankPathScore[path] = y[0]
    InitialPathsWithFinalBlank = {path}

    InitialPathsWithFinalSymbol = set()
    for i in range(len(SymbolSets)):
        path = SymbolSets[i]
        InitialPathScore[path] = y[i + 1]
        InitialPathsWithFinalSymbol.add(path)
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore


def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}

    for path in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

    for path in PathsWithTerminalSymbol:

        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] = UpdatedBlankPathScore[path] + PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}

    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSet)):
            newpath = path + SymbolSet[i]
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i + 1]

    for path in PathsWithTerminalSymbol:
        for i in range(len(SymbolSet)):
            newpath = path if (SymbolSet[i] == path[-1]) else path + SymbolSet[i]
            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath] = UpdatedPathScore[newpath] + PathScore[path] * y[i + 1]
            else:
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[i + 1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore, PrunedPathScore = {}, {}
    PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol = set(), set()
    scorelist = []

    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    scorelist.sort(reverse=True)
    cutoff = scorelist[BeamWidth] if (BeamWidth < len(scorelist)) else scorelist[-1]

    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]

    for p in PathsWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore

    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] = FinalPathScore[p] + BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        # Follow the pseudocode from lecture to complete beam search :-)
        PathScore = {}
        BlankPathScore = {}
        num_symbols, seq_len, batch_size = y_probs.shape

        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(self.symbol_set,
                                                                                                                 y_probs[:,
                                                                                                                 0, :])

        for t in range(1, seq_len):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank,
                                                                                               NewPathsWithTerminalSymbol,
                                                                                               NewBlankPathScore,
                                                                                               NewPathScore,
                                                                                               self.beam_width)

            NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol,
                                                                           y_probs[:, t, :], BlankPathScore, PathScore)

            NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol,
                                                                        self.symbol_set, y_probs[:, t, :], BlankPathScore,
                                                                        PathScore)

        MergedPaths, mergedPathScores = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
                                                            NewBlankPathScore, NewPathScore)

        bestPath = max(mergedPathScores, key=mergedPathScores.get)
        return (bestPath, mergedPathScores)