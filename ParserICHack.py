import dateparser as dp
import string
import numpy as np
import networkx as nx
import spacy
from spacy_hunspell import spaCyHunSpell


class ParserICHack():

    def __init__(self):
        self.nlp = spacy.load('en')
        self.hunspell = spaCyHunSpell(self.nlp, ('./src/hunspell/en_US.dic', './src/hunspell/en_US.aff'))
        self.nlp.add_pipe(self.hunspell)
        self.nlp.add_pipe(self.merge_entities, name='merge_entities')
        self.exclude = set(string.punctuation)

    def merge_entities(self, doc):
        """Preprocess a spaCy doc, merging entities into a single token.
        Best used with nlp.add_pipe(merge_entities).

        doc (spacy.tokens.Doc): The Doc object.
        RETURNS (Doc): The Doc object with merged noun entities.
        """
        spans = [(e.start_char, e.end_char, e.root.tag, e.root.dep, e.label)
                 for e in doc.ents]
        for start, end, tag, dep, ent_type in spans:
            doc.merge(start, end, tag=tag, dep=dep, ent_type=ent_type)
        return doc

    def tokenize(self, sentence):
        tokenized_sentence = []
        for token in sentence.split(' '): # simplest split is
            token = ''.join(ch for ch in token if ch not in self.exclude)
            if token != '':
                tokenized_sentence.append(token.lower())
        return ' '.join(tokenized_sentence)

    def correction_spacy(self, parsed):
        corrected = []
        for w in parsed.doc:
            if not(w.text in self.exclude) and not(w._.hunspell_spell):
                if len(w._.hunspell_suggest)>0 :
                    corrected.append(w._.hunspell_suggest[0])
                else:
                    corrected.append(w.text)
            else:
                corrected.append(str(w))
        return self.nlp(' '.join(corrected))

    def get_date_sentence(self, sentence):
        date_in_sentence = []
        for entity in sentence.ents:
            if entity.label_=="DATE":
                date_in_sentence.append(entity.text)
        return date_in_sentence

    # Date extraction
    def time_convert(self, list_str_date):
        return [dp.parse(w) for w in list_str_date]

    def get_attr(self, sentence, attribut):
        nsubj = []
        for token in sentence:
            if token.dep_ == attribut:
                nsubj.append(token)
        return nsubj

    def get_pos(self, sentence, pos):
        nsubj = []
        for token in sentence:
            # print(token.text, token.pos_, token.dep_)
            if token.pos_ == pos:
                nsubj.append(token)
        return nsubj

    def get_dep_graph(self, document):
        edges = []
        for token in document:
            # FYI https://spacy.io/docs/api/token
            for child in token.children:
                edges.append(('{0}'.format(token),
                              '{0}'.format(child)))

        return nx.Graph(edges)

    def get_closest_relation(self, graph, first_class, second_class):
        paths =[[(elt2.text, elt1) for elt2 in second_class ] for elt1 in first_class]
        lengths =[[nx.shortest_path_length(graph, source=subj.text, target=date) for subj in second_class ] for date in first_class]

        index_min = [np.argmin(length) for length in lengths]
        subject_date = []
        for i, date in enumerate(first_class):
            subject_date.append(paths[i][index_min[i]])

        return subject_date

    def word_is_negated(self, word):
        """ """

        for child in word.children:
            if child.dep_ == 'neg':
                return True

        if word.pos_ in {'VERB'}:
            for ancestor in word.ancestors:
                if ancestor.pos_ in {'VERB'}:
                    for child2 in ancestor.children:
                        if child2.dep_ == 'neg':
                            return True

        return False

    def find_negated_wordSentIdxs_in_sent(self, sent, idxs_of_interest=None):
        """ """

        negated_word_idxs = set()
        for word_sent_idx, word in enumerate(sent):
            if idxs_of_interest:
                if word_sent_idx not in idxs_of_interest:
                    continue
            if self.word_is_negated(word):
                negated_word_idxs.add(word_sent_idx)

        return negated_word_idxs

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def extract(self, input_text):
        ## First Task, simple cleaning for preparing proper tokenize
        # input_text = tokenize(input_text)

        ## Next, the text is parse with nlp from Spacy
        parsed = self.nlp(input_text)
        corrected = self.correction_spacy(parsed)

        pos = []
        neg = []
        pos_str = []
        neg_str = []
        for sentence in corrected.sents:
            # Spellcheck
            parse_sent = self.nlp(sentence.text)
            # Date extraction (as string)
            dates_str = self.get_date_sentence(parse_sent)
            # Extraction of subject and verbs (as token)
            subjects = self.get_attr(parse_sent, "nsubj")
            verbs = self.get_pos(parse_sent, "VERB")

            # Get the graphs of dependancy (as nx graphs)
            graph = self.get_dep_graph(parse_sent)
            # Get the closest subjects and verbs
            subj_date = self.get_closest_relation(graph, dates_str, subjects)
            verb_date = self.get_closest_relation(graph, dates_str, verbs)
            # Get all the negatives terms of the sentences
            negatives = [parse_sent[negate].text for negate in self.find_negated_wordSentIdxs_in_sent(parse_sent)]

            negative_time = [item[1] for item in verb_date if (item[0] in negatives or item[1] in negatives)]
            positive_time = [item[1] for item in verb_date if not(item[1] in negative_time)]

            pos_str += positive_time
            neg_str += negative_time
            neg += [ (key[0], dp.parse(key[1])) for key in subj_date if key[1] in negative_time]
            pos += [ (key[0], dp.parse(key[1])) for key in subj_date if key[1] in positive_time]

        return pos, neg, pos_str, neg_str


