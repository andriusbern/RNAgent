import rlfold.settings as settings
import os, html
import fileinput
import sys, webbrowser
from selenium import webdriver


from tempfile import mkstemp
from shutil import move
from os import fdopen, remove

def replace(file_path, pattern, subst, num=0):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            counter = -1
            for line in old_file:
                if pattern in line:
                    counter += 1
                if pattern in line and counter == num:
                    new_file.write(line.replace(pattern, subst))
                else:
                    new_file.write(line)
    remove(file_path)
    move(abs_path, file_path)

def modify_html(dotbr, seq=None, num=0, html='double'):
    filename = os.path.join(settings.MAIN_DIR, 'interface', 'fornac', 'dist', '{}.html'.format(html))

    # print(filename)

    with open(filename) as f:
        counter = -1 
        lines = f.readlines()
        for line in lines:
            if "'structure'" in line:
                counter += 1
                if counter == num:
                    olddotbr = line.split("'")[-2]
        counter = -1
        for line in lines:
            if 'sequence' in line:
                counter += 1    
                if counter == num:
                    oldseq = line.split("'")[-2]

    replace(filename, olddotbr, dotbr)
    if seq is not None:
        replace(filename, oldseq, seq)

    return filename

def show_rna(dotbr, seq=None, driver=None, num=0, html='double'):
    html = modify_html(dotbr, seq, num, html)
    driver.refresh()

def create_browser(html, browser='Chrome'):
    """
    Opens a browser window and loads a html from ../rlfold/interface/fornac/dist/*
    """
    if browser == 'Chrome':
        driver = webdriver.Chrome()
    if browser == 'Firefox':
        driver = webdriver.Firefox()
    
    path = os.path.join(settings.MAIN_DIR, 'interface', 'fornac', 'dist', html+'.html')
    driver.get('file://' + path)

    return driver
