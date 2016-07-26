# -*- coding: utf-8 -*-

import requests
import json

w2vUrl = "http://192.168.1.126:8085/cuservice/rest/word2vec/sentencevec?src="

wvDict = {
    # Badcases: 0722
    # "./Badcases/Single.txt": "./BadcasesVec/Single.vec",
    # Editorial: 0722
    # "./Editorial/Breakup0622-2327.csv": "./EditorialVec/Breakup0622-2327.vec",
    # "./Editorial/Crush0622-391.csv": "./EditorialVec/Crush0622-391.vec",
    # "./Editorial/Divorce0622-46.csv": "./EditorialVec/Divorce0622-46.vec",
    # "./Editorial/Love0622-2100.csv": "./EditorialVec/Love0622-2100.vec",
    # "./Editorial/Marriage0622-715.csv": "./EditorialVec/Marriage0622-715.vec",
    # "./Editorial/NotA0622-4221.csv": "./EditorialVec/NotA0622-4221.vec",
    # "./Editorial/NotDBA0622-16.csv": "./EditorialVec/NotDBA0622-16.vec",
    # "./Editorial/NotDBQ0622-3604.csv": "./EditorialVec/NotDBQ0622-3604.vec",
    # "./Editorial/NotQ0622-3051.csv": "./EditorialVec/NotQ0622-3051.vec",
    # "./Editorial/Single0622-615.csv": "./EditorialVec/Single0622-615.vec",
    # PM: 0722
    # "./PM/Breakup0623-1079.csv": "./PMVec/Breakup0623-1079.vec",
    # "./PM/Crush0623-850.csv": "./PMVec/Crush0623-850.vec",
    # "./PM/Divorce0623-750.csv": "./PMVec/Divorce0623-750.vec",
    # "./PM/Love0623-1251.csv": "./PMVec/Love0623-1251.vec",
    # "./PM/Marriage0623-906.csv": "./PMVec/Marriage0623-906.vec",
    # "./PM/Single0623-843.csv": "./PMVec/Single0623-843.vec",
    "./PM/Crush0725-602.csv" : "./PMVec/Crush0725-602.csv",
    "./PM/Divorce0725-600.csv" : "./PMVec/Divorce0725-600.csv",
    "./PM/Single0725-606.csv" : "./PMVec/Single0725-606.csv"
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
                print resW2v
                vf.write(str(resW2v).strip('[').strip(']') + "\n")

# wFile = "./csv/not_q.csv"
# vFile = "./vec/not_q.vec"
