# -*- coding: utf-8 -*-

import requests
import json

w2vUrl = "http://192.168.1.27:8080/conversation/rest/word2vec/sentencevec?src="

wFile = "./csv/not_q.csv"
vFile = "./vec/not_q.vec"

with open(wFile) as wf:
    with open(vFile, 'w') as vf:
        for line in wf:
            sent = line.strip()
            w2vSent = w2vUrl + sent
            res = requests.get(w2vSent)
            resText = res.text.strip().replace('\n', '\\n').replace('\r', '')
            resJson = json.loads(resText)
            resW2v = resJson.get("sentenceVector")
            print resW2v
            vf.write(str(resW2v) + "\n")
