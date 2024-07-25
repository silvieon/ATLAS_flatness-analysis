Hello! This is the text documentation for the OGP Flatness Analysis code. 

1. HOW TO USE:

--> open the file. This should open VS code. 

--> run the file by clicking the play button (should look somewhat like |>) in the top right corner. 
    This will open a new window that looks vaguely like something straight out of Windows XP. 

--> click the "Browse Files" button near the top of the new window. 
    This will take you to a page where you can look for files. 
    Navigate to the folder where flatness data is stored and select a flatness data file. 

****IF THE ANALYSIS IS OF DATA NOT FROM A DEFECT MATRIX MEASUREMENT****

--> click the button to analyze. 
    There will be blue text at the top of the screen. 

*** If the text says "multi-module file split into single-module files" or something of the sort, browse files again. 
    The code has created a folder next to the file you initially selected. 
    In this folder is data files that contain single module region's flatness data. 

*** If the text says something along the lines of "saved to _______________" analysis worked. 
    You should see graphs in the bottom of the window... 
        and the colored indicator in the middle should show whether the module region is in spec or not.
    You can navigate to the filepath (the string of characters in the ___________) 
        and do whatever you want the analysis graphs and text outputs from there. 

****IF THE ANALYSIS IS OF DEFECT MATRIX DATA****

--> do a "first pass" analysis of the file if you've just gotten the file out of the OGP. 
    This is the same process by which the file gets split in the non-defect matrix case, just with only one file as output. 

--> find the folder containing the results from the stave core module region (non-defect matrix) measurement. 
    There will be a .txt file titled "overview".

--> inside the "overview" file, copy the section of text specifying the normal vector.
    It should look like "normal=Vector" blah blah blah. copy the numbers. 
    It'll be fine if you get the parentheses and brackets, just try not to get any letters.

--> in the analysis window, enter the text you just copied into the box labeled "norm. vec. of fit plane".

--> click the box labeled "defect matrix", and then hit "analyze". 

--> the process to find the file is the same as stated above. 
    Your results visualizations are also in the same place. 


****NOTES****
The "enforce z-min" checkbox is built in to avoid the analysis getting derailed by the unfortunate possibility that the OGP accidentally measured off the side of the stave core. 
It shouldn't be _strictly_ necessary but it's probably good to keep turned on. 
If you're getting wonky results that might be the culprit; it's unlikely but turning it off for a measurement shouldn't break anything. 



~Silvia Wang, Tipton Group --- ATLAS ITK project at Yale Wright Labs. July 2024