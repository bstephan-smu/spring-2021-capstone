import en_core_web_sm
import re
import nltk

# split number word combinations
def split_numbers(self, word_list):
    output = []
    word_number_sequences = "([0-9])([a-z]{1,})"
    for row in word_list:
        row_list = []
        for word in row.split(" "):
            # print(word)
            extraction = re.search(word_number_sequences, word)
            if extraction is not None:
                row_list.extend([extraction.group(1), extraction.group(2)])
                # output.extend(extraction.split(" "))
            else:
                row_list.append(word)
        output.append(' '.join(row_list))
    return output


# lookup words from custom lexicon
def custom_lexicon(self, word_list):
    reason_for_visit_lexicon = {
        "mos": "month",
        "yrl": "annual",
        "yrly": "annual",
        "yearly": "annual",
        "mo": "month",
        "months": "month",
        "mnth": "month",
        "mth": "month",
        "mon": "month",
        "fu": "",  # followup is sort of needless information here
        "f/u": "",
        "f/up": "",
        "wk": "week",
        "wks": "week",
        "w": "week",
        "m": "month",
        "meds": "medications",
        "med": "medications",
        "weeks": "week",
        "np": "new patient",
        "inr": "international normalized ratio",
        "bp": "blood pressure",
        "htn": "high blood pressure",
        "hypertension": "high blood pressure",
        "r": "right",
        "l": "left",
        "lvm": "left ventricular mass",
        "dx": "diagnosis",
        "w/": "with",
        "appt": "appointment",
        "pcp": "primary care provider",
        "dm": "diabetes",
        "uti": "urinary tract infection",
        "ms": "multiple sclerosis",
        "ep": "electrophysiology (heart activity assessment)"
    }
    output = []
    for row in word_list:
        row_list = []
        words = row.split(" ")
        for word_ in words:
            if word_ in reason_for_visit_lexicon.keys():
                word_ = reason_for_visit_lexicon.get(word_)
            else:
                word_ = word_
            row_list.append(word_)
        # row_list.extend(words_list)
        output.append(' '.join(row_list))
    return output
    # make sure to split lexicon definition by space and strip any non alpha numeric characters


def strip_non_alpha(word_list):
    output = []
    for row in word_list:
        row_list = []
        for word in row.split(" "):
            search_ = re.search("[a-z0-9]{1,}", word)
            if search_ is not None:
                word_ = search_.group(0)
            else:
                word_ = word
            row_list.append(word_)
        output.append(' '.join(row_list))
    return output


def porter_stemmer(word_list):
    ps = nltk.PorterStemmer()
    output = []
    for row in word_list:
        row_list = []
        for word in row.split(" "):
            word_ = ps.stem(word)
            row_list.append(word_)
        output.append(' '.join(row_list))
    return output

# Get noun phrases
nlp = en_core_web_sm.load()

def getNounChunks(text_data):
    doc = nlp(text_data)
    noun_chunks = list(doc.noun_chunks)
    noun_chunks_strlist = [chunk.text for chunk in noun_chunks]
    noun_chunks_str = '_'.join(noun_chunks_strlist)
    return noun_chunks_str