# Preprocessing Notes
1. Download latex files from KDD Cup 2003 dataset.
2. Parase the files and extract suitable latex math equations (formulas from now on)
3. Normalize the extracted formulas by parsing them using katex parser and then regenerating latex from the parse-tree.
4. Render formulas to jpeg images (using pdflatex)
5. Create a pandas dataframe mapping images to normalized formula
6. Clean formula text
7. Tokenize formulas and extract vocabulary. Clean vocabulary and bad formulas.
8. Analyze data and remove very large images and latex sequences (in order to limit image size)
9. ...
