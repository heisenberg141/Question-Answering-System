import nltk
import sys
import string
import os
import math
import operator

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    corpus=dict()
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory,file)) as f:
                corpus[file]=f.read()
    return corpus
    raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    wordlist=list()
    document=document.lower()
    wordlist=nltk.word_tokenize(document)
    wordlist=[word for word in wordlist if word not in nltk.corpus.stopwords.words("english") and not all(char in string.punctuation for char in word)]
    # print(wordlist)
    return wordlist		
    raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    TotalDocuments=len(documents)
    # print(len(documents))
    wordCount=dict()
    
    for doc in documents:
        for word in documents[doc]:
            wordCount[word]=0

    for word in wordCount:
        count=0
        for doc in documents:
            if word in documents[doc]:
                count+=1
        wordCount[word]=count
    IDF=dict()
    # IDF=sorted(IDF,key=IDF.values(),reverse=True)
    for word in wordCount:
        IDF[word]=math.log(TotalDocuments/wordCount[word])
    
    return IDF
    raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    file_order=dict()
    for file in files:
        file_order[file]=0

    for file in files:
        for word in query:
            if word in files[file]:
                file_order[file]+=files[file].count(word)*idfs[word]
            else:
                file_order[file]+=0
    file_order=dict(sorted(file_order.items(),key=operator.itemgetter(1),reverse=True))
    # print(file_order)
    # print(list(file_order.keys()))
    return list(file_order.keys())[:n]    

    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rank=list()
    for sentence in sentences:
        sentence_values = [sentence, 0, 0]

        for word in query:
            if word in sentences[sentence]:
                sentence_values[1] += idfs[word]
                sentence_values[2] += sentences[sentence].count(word) / len(sentences[sentence])

        rank.append(sentence_values)
        
    return [sentence for sentence, word_measure, query_term_dens in sorted(rank, key=lambda item: (item[1], item[2]), reverse=True)][:n]

if __name__ == "__main__":
    main()
