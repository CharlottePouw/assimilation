import glob
import os
import re

data_dir = "/home/charlotte/Documents/LibriSpeech"
chapters = data_dir + "/CHAPTERS.TXT"

### EXTRACT UTF-8 BOOKS ###

all_text = []
book_path = data_dir +"/books/utf-8"

with open(chapters, 'r') as infile:
    lines = infile.readlines()

target_books = set()
for line in lines:
    if not line.startswith(';'):
        components = line.split('|')
        if 'train' in components[3]:
            target_books.add(components[5].strip())

for book_id in list(target_books):
    if os.path.exists(f"{book_path}/{book_id}/{book_id}.txt.utf-8"):
        with open(f"{book_path}/{book_id}/{book_id}.txt.utf-8", 'r', encoding='utf-8') as infile:
            content = infile.read()
            content_clean = content.replace("\n", " ")
            all_text.append(content_clean)

### EXTRACT ASCII BOOKS ###

book_path = data_dir +"/books/ascii"

for book_id in list(target_books):
    if os.path.exists(f"{book_path}/{book_id}/{book_id}.txt"):
        with open(f"{book_path}/{book_id}/{book_id}.txt", 'r', encoding='ascii', errors='ignore') as infile:
            content = infile.read()
            content_clean = content.replace("\n", " ")
            all_text.append(content_clean)

text = " ".join(all_text).lower()

print("clean pan/spoon/fork:")
print(len(re.findall(r'clean pan', text)))
print(len(re.findall(r'clean spoon', text)))
print(len(re.findall(r'clean fork', text)))

print("own plan/choice/life:")
print(len(re.findall(r'own plan', text)))
print(len(re.findall(r'own choice', text)))
print(len(re.findall(r'own life', text)))

print("tan belt/shirt/scarf:")
print(len(re.findall(r'tan belt', text)))
print(len(re.findall(r'tan shirt', text)))
print(len(re.findall(r'tan scarf', text)))

print("lean back/shape/line:")
print(len(re.findall(r'lean back', text)))
print(len(re.findall(r'lean shape', text)))
print(len(re.findall(r'lean line', text)))

print("thin packet/leaflet/notebook:")
print(len(re.findall(r'thin packet', text)))
print(len(re.findall(r'thin leaflet', text)))
print(len(re.findall(r'thin notebook', text)))

print("fat puppy/squirrel/monkey")
print(len(re.findall(r'fat puppy', text)))
print(len(re.findall(r'fat squirrel', text)))
print(len(re.findall(r'fat monkey', text)))

print("wet pants/socks/shoes")
print(len(re.findall(r'wet pants', text)))
print(len(re.findall(r'wet socks', text)))
print(len(re.findall(r'wet shoes', text)))

print("great cruise/match/fight")
print(len(re.findall(r'great cruise', text)))
print(len(re.findall(r'great match', text)))
print(len(re.findall(r'great fight', text)))

print("sweet cocktail/liquor/chocolate")
print(len(re.findall(r'sweet cocktail', text)))
print(len(re.findall(r'sweet liquor', text)))
print(len(re.findall(r'sweet chocolate', text)))

print("green cup/chair/vase")
#print(len(re.findall(r'\bgreen\b', text)))
print(len(re.findall(r'green cup', text)))
print(len(re.findall(r'green chair', text)))
print(len(re.findall(r'green vase', text)))

print("plain condoes/churches/chapels")
#print(len(re.findall(r'\bplain\b', text)))
print(len(re.findall(r'plain condoes', text)))
print(len(re.findall(r'plain churches', text)))
print(len(re.findall(r'plain chapels', text)))

print("fun game/night/day")
#print(len(re.findall(r'\bfun\b', text)))
print(len(re.findall(r'fun game', text)))
print(len(re.findall(r'fun night', text)))
print(len(re.findall(r'fun day', text)))

print("mad brother/daughter/mother")
#print(len(re.findall(r'\bmad\b', text)))
print(len(re.findall(r'mad brother', text)))
print(len(re.findall(r'mad daughter', text)))
print(len(re.findall(r'mad mother', text)))

print("sad ballet/novel/movie")
#print(len(re.findall(r'\bsad\b', text)))
print(len(re.findall(r'sad ballet', text)))
print(len(re.findall(r'sad novel', text)))
print(len(re.findall(r'sad movie', text)))

print("bad beer/lunch/dish")
#print(len(re.findall(r'\bbad\b', text)))
print(len(re.findall(r'bad beer', text)))
print(len(re.findall(r'bad lunch', text)))
print(len(re.findall(r'bad dish', text)))

print("red glasses/lipstick/necklace")
#print(len(re.findall(r'\bred\b', text)))
print(len(re.findall(r'red glasses', text)))
print(len(re.findall(r'red lipstick', text)))
print(len(re.findall(r'red necklace', text)))
