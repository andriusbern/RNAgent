from rlif.settings import ConfigManager as settings
import os, html
import fileinput
import sys, webbrowser
# import imgkit
# from PIL import Image
import numpy as np
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
    filename = os.path.join(settings.DISPLAY, '{}.html'.format(html))
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
    if driver is not None:
        driver.refresh()

def create_browser(html, browser='Chrome'):
    """
    Opens a browser window and loads a html from
    """
    from selenium import webdriver
    if browser == 'Chrome':
        driver = webdriver.Chrome()
    if browser == 'Firefox':
        driver = webdriver.Firefox()
    
    path = os.path.join(settings.MAIN_DIR, 'display', html+'.html')
    driver.get('file://' + path)

    return driver

def draw_rna(sequence):
    options = {'quiet': ''}
    show_rna(sequence, seq=None, driver=None, num=0, html='rna_disp')
    out = os.path.join(settings.RESULTS, 'images', 'rna.jpg')
    imgkit.from_file(os.path.join(settings.DISPLAY, 'rna_disp.html'), out, options=options)
    img = np.array(Image.open(out))

    return img