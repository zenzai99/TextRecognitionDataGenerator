import argparse
import os, errno
import random
import string
import sys
import random
import shutil
import pandas as pd

from tqdm import tqdm
from string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly
)
from data_generator import FakeTextDataGenerator
from multiprocessing import Pool

def margins(margin):
    margins = margin.split(',')
    if len(margins) == 1:
        return [int(margins[0])] * 4
    return [int(m) for m in margins]

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="out/",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="en"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images if horizontal, else the width",
        default=32,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures, 4:Random color 5: random 1 to 4",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings, 3: [ID].[EXT] + Report file .csv",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=1
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=0
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format., Use rnd: Random RGB 0 to 255, rndInList: random color in colorList " ,
        default='#282828',
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=1.0
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(5, 5, 5, 5)
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        help="Apply a tight crop around the rendered text",
        default=False
    )
    parser.add_argument(
        "-ft",
        "--font",
        type=str,
        nargs="?",
        help="Define font to be used"
    )
    parser.add_argument(
        "-rbs",
        "--random_blur_and_skew",
        type=bool,
        nargs="?",
        help="random_blur_and_skew",
        default=False
    )
    parser.add_argument(
        "-bcm",
        "--background_color_mode",
        type=str,
        nargs="?",
        help="random_background_color_mode use rnd: Random RGB 0 to 255, rndInList: random color in colorList ",
        default="rnd"
    )
    parser.add_argument(
        "-rfs",
        "--random_font_size",
        type=bool,
        nargs="?",
        help="random_font_size True: random font size in ratio list ",
        default=False
    )
    
    return parser.parse_args()


def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict

def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """

    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    elif lang == 'th':
        return [os.path.join('fonts/tha', font) for font in os.listdir('fonts/tha')]
    else:
        return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]

colorList =[(255,0,0,1),
           (0,255,0,1),
           (0,0,255,1),
           (255,255,0,1),
           (0,255,255,1),
           (255,0,255,1),
           (255,255,255,1),
           (0,0,0,1)]

percentRatioList = [5,10,20,30,50,70,80,90,100]

# Random text color in list
from colormap import rgb2hex
def RandomTextColorInList(count: int):
    colorListRGB = []
    for i in range(count):
        rnd = random.randint(0,len(colorList)-1)
        color = colorList[rnd]
        colorListRGB.append(color)
    
    # Convert to hex colors
    colorListHex = []
    for i,e in enumerate(colorListRGB):
        hexColor = rgb2hex(e[0],e[1],e[2])
        colorListHex.append(hexColor)
        
    return colorListHex

# Random text color
from colormap import rgb2hex
def RandomTextColor(count: int):
    colorListRGB = []
    for i in range(count):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255),1)
        colorListRGB.append(color)
    
    # Convert to hex colors
    colorListHex = []
    for i,e in enumerate(colorListRGB):
        hexColor = rgb2hex(e[0],e[1],e[2])
        colorListHex.append(hexColor)
        
    return colorListHex

# Random background color in list
def RandomBackgroundColorInList(count: int)-> list:
    colorList = []
    for i in range(count):
        rnd = random.randint(0,len(colorList)-1)
        color = colorList[rnd]
        colorList.append(color)
        
    return colorList

# Random background color
def RandomBackgroundColor(count: int)-> list:
    colorList = []
    for i in range(count):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255),1)
        colorList.append(color)
        
    return colorList
             
# Create report .csv
def CreateReport(dataframe):
    df = pd.DataFrame(dataframe,columns=['File name','Text','Font','Font size ratio','Font color','Background','Image size','Distorsion','Blur','Skew'])
    df.to_csv('out/Report.csv',index = False)
    print("Report.csv is written")

def main():
    """
        Description: Main function
    """

    # Argument parsing
    args = parse_arguments()
    # Create the directory if it does not exist.
    try:
        if os.path.exists(args.output_dir) == True:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list
    lang_dict = load_dict(args.language)

    # Create font (path) list
    if not args.font:
        fonts = load_fonts(args.language)
    else:
        if os.path.isfile(args.font):
            fonts = [args.font]
        else:
            sys.exit("Cannot open font")

    # Creating synthetic sentences (or word)
    strings = []
    
    if args.use_wikipedia:
        strings = create_strings_from_wikipedia(args.length, args.count, args.language)
    elif args.input_file != '':
        strings = create_strings_from_file(args.input_file, args.count)
    elif args.random_sequences:
        strings = create_strings_randomly(args.length, args.random, args.count,
                                          args.include_letters, args.include_numbers, args.include_symbols, args.language)
        # Set a name format compatible with special characters automatically if they are used
        if args.include_symbols or True not in (args.include_letters, args.include_numbers, args.include_symbols):
            args.name_format = 2
    else:
        strings = create_strings_from_dict(args.length, args.random, args.count, lang_dict)

    string_count = len(strings)
    
    # Random BG color
    colorBGList = []
    
    backgroundList = []
    
    if args.background == 0 or args.background == 1 or args.background == 2 or args.background == 3:
        for i in range(args.count):
            colorBGList.append(args.background)
            backgroundList.append(args.background)
    elif args.background == 4:
        
        for i in range(args.count):
            if args.background_color_mode == "rndInList":
                colorBGList =  RandomBackgroundColorInList(args.count)
                backgroundList.append(4)
            elif args.background_color_mode == "rnd":
                colorBGList =  RandomBackgroundColor(args.count)
                backgroundList.append(4)
            
    elif args.background == 5:
        
        for i in range(args.count):
            args.background = random.randint(0,4)
            
            if args.background == 4:
                args.background = 4
                if args.background_color_mode == "rndInList":
                    colorBGList.append(colorList[random.randint(0,len(colorList)-1)])
                    backgroundList.append(4)
                    
                elif args.background_color_mode == "rnd":
                    colorBGList.append((random.randint(0,255),random.randint(0,255),random.randint(0,255),1))
                    backgroundList.append(4)
            else:
                rndBackground = random.randint(0,3)
                colorBGList.append(rndBackground)
                backgroundList.append(rndBackground)
    
    # Random text color
    if args.text_color == 'rndInList':
        colorTextList =  RandomTextColorInList(args.count)
    elif args.text_color == 'rnd':
        colorTextList =  RandomTextColor(args.count)
    else:
        colorTextList = []
        for i in range(args.count):
            colorTextList.append(args.text_color)
        
    # Random font
    fontList = []
    for i in range(args.count):
        fontList.append( fonts[random.randrange(0, len(fonts)-1)])
        
    # Distorsion list
    distorsionList = []
    if args.distorsion == 3:
        for i in range(args.count):
            distorsionList.append(random.randint(0,2))
    else:
        for i in range(args.count):
            distorsionList.append(args.distorsion)
    
    # Skew & Blur list
    blurList = []
    skewList = []
    if args.random_blur_and_skew == True:
        for i in range(args.count):
            blurList.append(random.choice([True, False]))
            skewList.append(random.choice([True, False]))
    else:
        for i in range(args.count): 
            blurList.append(args.random_blur)
            blurList.append(args.random_skew)
            
    # Font size List 
    fontSizeList = []
    ratioList = []
    
    for i in range(args.count):
        if args.random_font_size == True:
            ratio = percentRatioList[0] # random.randint(0,len(percentRatioList)-1)
            fontSizeList.append(ratio)
            ratioList.append(int(ratio))
        elif args.random_font_size == False:
            fontSizeList.append(int(args.format))
            ratioList.append(int(args.format/args.format*100))
    
    
    p = Pool(args.thread_count)
    for _ in tqdm(p.imap_unordered(
        FakeTextDataGenerator.generate_from_tuple,
        zip(
            [i for i in range(0, string_count)],
            strings,
            [e for e in fontList],
            [args.output_dir] * string_count,
            [args.format] * string_count,
            [args.extension] * string_count,
            [args.skew_angle] * string_count,
            [e for e in skewList],
            [args.blur] * string_count,
            [e for e in blurList],
            [e for e in backgroundList],
            [e for e in distorsionList],
            [args.distorsion_orientation] * string_count,
            [args.handwritten] * string_count,
            [args.name_format] * string_count,
            [args.width] * string_count,
            [args.alignment] * string_count,
            [e for e in colorTextList] * string_count,
            [args.orientation] * string_count,
            [args.space_width] * string_count,
            [args.margins] * string_count,
            [args.fit] * string_count,
            [e for e in colorBGList],
            [e for e in fontSizeList]
        )
    ), total=args.count):
        pass
    p.terminate()

    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(os.path.join(args.output_dir, "labels.txt"), 'w', encoding="utf-8") as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                f.write("{} {}\n".format(file_name, strings[i]))
                print(strings[i])
                
    elif args.name_format == 3:
        from PIL import Image
        dataframe = []
        for i in range(args.count):
            im = Image.open(args.output_dir+str(i)+'.jpg')
            fontName = fontList[i]
            
            
            if colorBGList[i] == 0:
                colorBGList[i] = "gaussian noise"
            elif colorBGList[i] == 1:
                colorBGList[i] = "plain white"
            elif colorBGList[i] == 2:
                colorBGList[i] = "quasicrystal"
            elif colorBGList[i] == 3:
                colorBGList[i] = "picture"
            elif type(colorBGList[i]) == type((0,0,0,0)):
                colorBGList[i] = "color background " + str(colorBGList[i]) 
                
            if distorsionList[i] == 0:
                distorsionList[i] = "None (Default)"
            elif distorsionList[i] == 1:
                distorsionList[i] = "Sine wave"
            elif distorsionList[i] == 2:
                distorsionList[i] = "Cosine wave"
            
            tupleData = (i,strings[i],fontName[10:],str(ratioList[i])+"%",colorTextList[i],colorBGList[i],im.size,distorsionList[i],blurList[i],skewList[i])
            
            dataframe.append(tupleData)
            print(tupleData)
            
        CreateReport(dataframe)
        

if __name__ == '__main__':
    main()
