#!/usr/bin/env python
#_*_coding:utf-8_*_

import re, os, sys

def readFasta(file):
	myFasta = []

	with open(file, 'r') as fp2:
		for (m, line1) in enumerate(fp2):
			now = line1.split(',')
			protein_name = now[1]
			sequence = now[0]
			myFasta.append([protein_name, sequence])
	print(myFasta)
	return myFasta
