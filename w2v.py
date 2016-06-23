# -*- coding: utf-8 -*-

import requests
import json

w2vUrl = "http://192.168.1.27:8080/conversation/rest/word2vec/sentencevec?src="

wvDict = {
    "./csv/Breakup0622-2327.csv": "./vec/Breakup0622-2327.vec",
    "./csv/Breakup0623-1079.csv": "./vec/Breakup0623-1079.vec",
    "./csv/Crush0622-391.csv": "./vec/Crush0622-391.vec",
    "./csv/Crush0623-850.csv": "./vec/Crush0623-850.vec",
    "./csv/Divorce0622-46.csv": "./vec/Divorce0622-46.vec",
    "./csv/Divorce0623-750.csv": "./vec/Divorce0623-750.vec",
    "./csv/Love0622-2100.csv": "./vec/Love0622-2100.vec",
    "./csv/Love0623-1251.csv": "./vec/Love0623-1251.vec",
    "./csv/Marriage0622-715.csv": "./vec/Marriage0622-715.vec",
    "./csv/Marriage0623-906.csv": "./vec/Marriage0623-906.vec",
    "./csv/NotA0622-4221.csv": "./vec/NotA0622-4221.vec",
    "./csv/NotDBA0622-16.csv": "./vec/NotDBA0622-16.vec",
    "./csv/NotDBQ0622-3604.csv": "./vec/NotDBQ0622-3604.vec",
    "./csv/NotQ0622-3051.csv": "./vec/NotQ0622-3051.vec",
    "./csv/Single0622-615.csv": "./vec/Single0622-615.vec",
    "./csv/Single0623-843.csv": "./vec/Single0623-843.vec"
}

for k, v in wvDict.iteritems():
    wFile = k
    vFile = v
    with open(wFile) as wf:
        with open(vFile, 'w') as vf:
            for line in wf:
                sent = line.strip()
                w2vSent = w2vUrl + sent
                res = requests.get(w2vSent)
                resText = res.text.strip().replace(
                    '\n', '\\n').replace('\r', '')
                resJson = json.loads(resText)
                resW2v = resJson.get("sentenceVector")
                # print resW2v
                vf.write(str(resW2v).strip('[').strip(']') + "\n")

# wFile = "./csv/not_q.csv"
# vFile = "./vec/not_q.vec"
