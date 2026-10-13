from annotated_text import annotated_text

from vphoberttagger.arguments import get_predict_argument
from vphoberttagger.predictor import ViTagger

import streamlit as st

args = get_predict_argument()
st.markdown("<h1 style='text-align: center;'>ViTagger Demo</h1>", unsafe_allow_html=True)
@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def load_models(model_path, no_cuda):
    tagger = ViTagger(model_path, no_cuda=no_cuda)
    return tagger
def process_text(doc, expected_tag):
    tokens = []
    color_map = {
        'CQ': '#d8f2fe',    # Cơ quan
        'ĐT': '#fef6ce',    # Đối tượng
        'VBPL': '#f4cfdb',  # Văn bản pháp luật
        'NG': '#edfcd9',    # Ngày tháng
        'SL': '#ffe2cc',    # Số lượng
        'PERSON': '#fef6ce',
        'PER': '#fef6ce',
        'ORGANIZATION': '#d8f2fe',
        'ORG': '#d8f2fe',
        'LOCATION': '#f4cfdb',
        'LOC': '#f4cfdb',
        'MISCELLANEOUS': '#edfcd9',
        'MISC': '#edfcd9',
    }
    for token in doc:
        tag = token[-1]
        if tag in expected_tag and tag != 'O':
            color = color_map.get(tag, "#e0e0e0")
            tokens.append((token[0], tag, color))
        else:
            tokens.append(" " + token[0] + " ")
    return tokens
tagger = load_models(args.model_path, args.no_cuda)
labels = []
for l in tagger.label2id:
    if l == 'O':
        continue
    else:
        _, tag = l.split("-")
        if tag not in labels:
            labels.append(tag)
tags = st.multiselect("Which label do you extract?", labels, default=labels)
text_input = st.text_area("Enter your text to extract")
if st.button('Extract'):
    doc = tagger.extract_entity_doc(text_input)
    tokens = process_text(doc, tags)
    annotated_text(*tokens)