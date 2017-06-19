# -*- coding:utf-8 -*-
import json
import os
import re

def extractor(target_file_name,source_file_name):
    targetFile = open(target_file_name, "w+",encoding= 'utf-8')
    for line in open(source_file_name, "r",encoding= 'utf-8'):
        proceeLine(line,targetFile)
        # targetFile.write(line_csv)

    targetFile.close()

def proceeLine(line_txt,target_file):
    is_print_header = True
    if '####' in line_txt:
        line_txt = line_txt.partition('####')[0]
    if '>>' in line_txt:
        line_txt = line_txt.partition('>>')[2]
    line_obj = json.loads(line_txt)
    for req in line_obj['network']:
        print(req)
        header = ''
        req_values = ''
        for k,v in req.items():
            if is_print_header:
                header += k+','
            req_values += v+','

        if is_print_header:
            target_file.write(header[:header.rindex(',')]+'\n')
            is_print_header = False
        target_file.write(req_values[:req_values.rindex(',')]+'\n')

if __name__== "__main__":
    extractor("data.csv","data_demo.txt")