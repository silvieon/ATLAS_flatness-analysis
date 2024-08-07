import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from tkinter import *
from tkinter import filedialog
from skspatial.objects import Points, Plane
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm, colors
import re
from itertools import compress

#variable to immediately store filepath of currently open file
global file
file = ""
#variable to store the directory location of currently open file--allows the "new analysis" button to open directory in the same location
global dirmemory
dirmemory = "/"

#opens and reads requested data source file
#command controlled by the "browse files" button
def browseFiles():
    global file 
    global dirmemory

    #sets the value of file
    file = filedialog.askopenfilename(initialdir = dirmemory, title = "Select a File", filetypes = (("Text files","*.txt*"),("all files","*.*")))
    
    #this is all to set the value of the file directory with a variable file name length
    dirmemory = os.path.dirname(file)

    #notification of action
    print("File explorer is open at location: " + dirmemory)
    label_file_explorer.configure(text="File Opened: " + file)

#controlled statement that will detect if the file consists of multiple module regions. 
#if the file is multiple modules' worth of data, it will split the multi-module file into single-module files. 
#command controlled by the "analyze" button
def confirmAnalysis():
    global file 
    plaintext = ""

    #front-facing error message for showing the operator that they have not yet selected a file
    if file == "":
        label_file_explorer.configure(text="You have not opened a file! Please select a file.")
    else:
        plaintext = readFile(file)
        print('creating path...')
        fileroot = file[0:len(file)-4]
        if not os.path.isdir(fileroot):
            os.makedirs(fileroot)
        print("path created successfully. ")
    
    #conditional statement. Decides whether the file is a multi-module file (contains the keyword "Plane")...
    #from there either splits the multi-module file into single-module files or analyzes the single-module file. 
    if plaintext.find("Plane   ") > -1: 
        splitFile(plaintext, fileroot)
    else:
        analyzeFile(plaintext, fileroot)
        file = ""

#this one just does what the name says. its actually that shrimple.
def readFile(file):
    #reading the file, regardless of file parsability, will require the operator to reset the UI if they want to pick a new file
    button_analyze.configure(relief=SUNKEN, command=NONE)
    button_explore.configure(relief=SUNKEN, command=NONE)
    button_reset.configure(relief=RAISED, command=reset)

    #this bit is intelligible just by reading it in the voice of someone who's, like, a <i>little<\i> drunk. 
    with open(file, "r") as text:
        plaintext = text.read()
    
    return plaintext

#splits multi-module data file into multiple single-module data files.
def splitFile(text, fileroot):
    fileName = os.path.basename(fileroot)
    #print(fileName)
    
    #uses the filename to find what to number each single-module file
    numberindex = fileName.find("-") - 2
    fileNumber = 0
    if not fileName[numberindex] == "1":
        fileNumber = int(fileName[numberindex + 1])
        #print(fileName[numberindex])
    else:
        fileNumber = int(fileName[numberindex:numberindex+2])

    #the while loop copies into each single-module file the plain text from the multi-module file with bounds: 
    #starting at either the beginning of the file or the first instance of the keyword "point" after any instance of the keyword "plane"
    #ending before the next instance of the keyword "plane"
    index = text.find("Plane   ")
    while index > -1:
        filename = fileroot + "/moduleLocation" + str(fileNumber) + ".txt"
        string = text[0:index]
        f = open(filename, "w")
        f.write(string)
        f.close()

        fileNumber += 1
        temp = text[index:len(text)]
        index2 = temp.find("Point   ")
        text = temp[index2:len(temp)]
        index = text.find("Plane   ")
        print("writing to " + filename + "...")
    #the product of this is multiple single-module files filled with only the data file header, "point" keywords, and associated positional data. 

    #outwards-facing notification of the file split being done. 
    print("split done.")
    label_file_explorer.configure(text="Multi-module data file split into single-module data files.")

#turns space-separated proprietary data in text file format into a Pandas dataframe
def reformatString(text):
    #searches for numbers. Each number in the proprietary text format bgins with "+" or "-". 
    nums = [text[i:i+10] for i in range(len(text)) if text.startswith("+", i) or text.startswith("-", i)]

    mask = [i[4:5]=="." and re.sub("[-+.]", "", i).isnumeric() for i in nums]

    nums = list(compress(nums, mask))

    #this bit separates the whole list of numbers into three lists of x,y,z values. 
    #since the original list goes down the line of the starting text format, we can know that the list of numbers goes in order: x,y,z,x,y,z,...
    numsMod0 = [nums[i] for i in range(len(nums)) if i%3 == 0]
    numsMod1 = [nums[i] for i in range(len(nums)) if i%3 == 1]
    numsMod2 = [nums[i] for i in range(len(nums)) if i%3 == 2]

    #this bit removes the 5 points at the beginning of every routine that are really close to the origin. 
    #also outputs a notification for these points being removed. 
    removeIndices = [not abs(float(i)) < 0.001 for i in numsMod1]
    if not defect.get():
        numsMod0 = list(compress(numsMod0, removeIndices))
        numsMod1 = list(compress(numsMod1, removeIndices))
        numsMod2 = list(compress(numsMod2, removeIndices))

    #this bit throws an error and exits the function if there is an unequal number of x,y,z values. 
    if not len(numsMod0) == len(numsMod1) == len(numsMod2):
        print("list 1 length: " + str(len(numsMod0)))
        print("list 2 length: " + str(len(numsMod1)))
        print("list 3 length: " + str(len(numsMod2)))
        print("these are not the same number. this is bad.")
        label_file_explorer.configure(text="Oh no! We've run into an error. Check data file formatting.")
        return None

    #this creates and returns a Pandas DataFrame that matches each x, y, or z value with corresponding x,y,z values to create 3d coordinates as expressed in OGP output data. 
    df = pd.DataFrame({'X/R Location': numsMod0, 'Y/A Location': numsMod1, 'Z Location': numsMod2})
    return df

#using x,y,z positional data, this function removes anything that is unreasonably below the median z-value of the distribution of points.
def minimumZ_val(dataframe):
    #notification of function activation
    print("removing outliers...")

    #turns x,y,z postional data in Pandas DataFrame into a z-value only array, with indices corresponding between the two formats.
    z_vals = np.array(dataframe.iloc[:,2], dtype=float)
    z_med = np.median(z_vals)

    #array of indices at which the "distance from median" value is large and the z-value is below the median. 
    #in effect, an array of outlier indices.
    mask = [i for i in range(len(z_vals)) if abs(z_med - z_vals[i]) > 0.25]

    #notifies operator of each point deemed an outlier and removed and its associated "distance from median" value.
    for i in mask:
        print("removed! z-difference: " + str(abs(z_med - z_vals[i])))

    #notifies operator if there are no outliers in dataset (no points removed).
    if len(mask) == 0:
        print("woah! no outliers to remove. \nmoving on...")
    
    removed = dataframe.drop(index=mask)
    return removed

#its really, unironically that shrimple. 
def analyzeFile(plaintext, fileroot):
    #this bit does what the print statement says it does. The directory stores exports of data analysis. 
    #print("making directory...")
    #fileroot = file[0:len(file)-4]
    #if not os.path.isdir(fileroot):
    #    os.makedirs(fileroot)
    #superceded by copy of code in "confirmAnalysis" method. 

    #this turns the plain text data into a numpy array of 3-d coordinates. 
    print("getting data...")
    data = reformatString(plaintext)

    #print(data)

    #print("length of data before outliers: " + str(len(data)))

        #if the "remove outliers" box is checked, the data will be scanned for outliers and the outlier points will be removed. 
        #the exact process is of questionable validity, but it works. 
    if zMinimum.get():
        data = minimumZ_val(data)
    data = data.to_numpy(dtype=float)

    #print("length of data after outliers: " + str(len(data)))

    #the code will then analyze the 3-d coordinates through sorting, visualization, and exporting.
    print("analyzing, doo de doo dooo~")

    #a MAGICAL set of methods imported from scikit-spatial that creates a best-fit plane from given 3-d data points. 
    #it uses single value decomposition? I think. the documentation is here: 
    #https://scikit-spatial.readthedocs.io/en/stable/api_reference/Plane/methods/skspatial.objects.Plane.best_fit.html
    p = Points(data)
    if defect.get():
        baseCandidates = [i for i in data if abs(i[0]) < 0.001 and abs(i[1]) < 0.001]
        basePoint = baseCandidates[0]

        nonDigitNumerical = [' ', '.', 'e', 'E', '-', '+']
        normal = norm.get()[norm.get().find("["):]
        vec = ''.join(c for c in normal if c.isdigit() or c in nonDigitNumerical)
        vector = [float(i) for i in vec.split()]

        plane = Plane(point = basePoint, normal=vector)
    else:
        plane = Plane.best_fit(p)
    #print(plane)

    #uses another bit of magic from scikit-spatial to determine which, if any, of the "peak" and "out of spec" categories each 3d point belongs in. 
    #again, documentation is here: 
    #https://scikit-spatial.readthedocs.io/en/stable/api_reference/Plane/methods/skspatial.objects.Plane.distance_point_signed.html
    #this bit also changes the "data" array from being a collection of 3d points to being a collection of 
    #   points on the x-y plane and each point's associated offset from the best fit plane. 
    for i in range(len(data)): 
        point = p[i]
        planeDistance = plane.distance_point_signed(point)
        data[i, 2] = planeDistance

    #creating arrays to store point info for peaks and points out of spec
    maxPoint = [i for i in data if i[2] == max(data[:,2])]
    minPoint = [i for i in data if i[2] == min(data[:,2])]
    eccentricPoints = [i for i in data if abs(i[2]) >= 75/1000]

    peakPoints = np.array([maxPoint[0], minPoint[0]])
    eccentricPoints = np.array(eccentricPoints)
    #print(peakPoints)
    #print(eccentricPoints)

    #histogram, heatmap, and residualsPlot are visualization functions that also export the visualization. 
    histogram(data, fileroot)
    heatmap(data, eccentricPoints, peakPoints, fileroot)
    residualsPlot(data, eccentricPoints, peakPoints, fileroot)

    #to_CSV and text_label are export-only functions that generate files making it easier to compare and summarize the data.
    to_CSV_whileLoop(data, fileroot)
    text_label(data, plane, peakPoints, fileroot)

    #labelConfigure alters the GUI label based on inputted data to quickly see if the stave core is usable at a glance.
    if defect.get():
        labelConfigure_defect(data)
    else:
        labelConfigure(eccentricPoints)

    #outward-facing notifications that allow operator to see that analysis is done and to know where to look for exported files. 
    label_file_explorer.configure(text="Saved figures to folder: "+ fileroot)
    print("done.")

#changes the label in the middle of the GUI to show through both words and color whether the stave has passed, failed, or needs retesting.
def labelConfigure(eccentricPoints):
    #encoding the state of the output in a truth value
    cap = False
    dip = False
    for i in eccentricPoints:
        if i[2] < 0:
            dip = True
        if i[2] > 0:
            cap = True
    truth = int(dip) * 2 + int(cap)
    #encoding goes as follows: 
    #0: neither cap nor dip, 1: cap, 2: dip, 3: both cap and dip

    #directory of values used to change the label depending on value of truth
    match truth:
        case 3 | 1:
            textOut = "STAVE CORE FAILED."
            bgColor = "orange red"
            fgColor = "black"
            #this corresponds to the rejection notice
        case 2:
            textOut = "RETEST FROM FAIL POINT."
            bgColor = "gold"
            fgColor = "black"
            #this corresponds to the B-class stave core notice
        case 0: 
            textOut = "STAVE CORE PASSED."
            bgColor = "dark green"
            fgColor = "white"
            #this corresonds to the good stave core notice

    #statement that changes the label. 
    label_goodness.configure(text=textOut, bg = bgColor, fg = fgColor)

    print("label configuration done.")

def labelConfigure_defect(data):
    fail = False

    basePoint = []
    for i in data: 
        if abs(i[0]) < 0.001 and abs(i[1]) < 0.001:
            basePoint = i
            #print(basePoint)
    pointsUnder = [i for i in data if i[2] <= basePoint[2]]

    for i in pointsUnder:
        count = 0
        tail = np.array([i[0], i[1]])
        for j in pointsUnder:
            head = np.array([j[0], j[1]])
            distance = np.linalg.norm(tail - head)
            if 0 < abs(distance) <= 2:
                count += 1
        print("x: " + str(i[0]) + ". y: " + str(i[1]) + ". out-of-spec adjacent/identity point count : "  + str(count))
        if count >= 8:
            fail = True

    if fail:
        textOut = "STAVE CORE FAILED."
        bgColor = "orange red"
        fgColor = "black"
        #this corresponds to the rejection notice
    else:
        textOut = "B-CLASS STAVE CORE."
        bgColor = "gold"
        fgColor = "black"
        #this corresponds to the B-class stave core notice
        
    label_goodness.configure(text=textOut, bg = bgColor, fg = fgColor)

    print("label configuration done.")

#changes the 3-column table of x,y,z data into a 10x10 (most of the time) grid of z-values ordered by x and y values. 
def to_CSV_whileLoop(data, fileroot):
    file = fileroot + "/offsets.csv"

    temp = -1000
    index = 0
    array, nums = [], []

    #continually adds z-values to nums as each corresponding x-value grows. 
    #this continues until it reaches the largest x-value of its neighbors, which is the rightmost element in a row.
    #upon seeing the end of a row, the loop stops adding elements to nums (the row array) and appends the finished row to 
    #a larger wrapper array that contains multiple rows and acts as the structure for columns, creating a 2d grid of values.
    while index < len(data):
        x_val = data[index, 0]
        z_val = np.round(data[index, 2], 6)
        if abs(z_val) > 0.075:
            z_val = 'ERROR: ' + str(z_val)
        if x_val <= temp:
            array.append(nums)
            nums = []
        nums.append(z_val)
        temp = x_val
        index += 1
    array.append(nums)
    
    #converting the grid into saveable file format.
    csv = pd.DataFrame(array)
    csv.to_csv(file)

    #outwards notifications of the method finishing
    print("CSV data holds " + str(len(data)) + " points. ")
    print("CSV data saved to " + file + "\nnext...")

#creates a .txt file containing values useful for an overview of the stave.
def text_label(data, plane, peakPoints, fileroot):
    file = fileroot + "/overview.txt"

    z_list = data[:,2]
    #the standard deviation of plane-offset values
    stdev = np.std(z_list)
    stdev = stdev * 1000
    #the maximum and minimum plane-offset values
    max_z = peakPoints[0,2] * 1000
    min_z = peakPoints[1,2] * 1000
    #the peak-to-peak distance of the distribution of plane-offsets.
    ptp = abs(max_z - min_z)
    text = "OVERVIEW. \nSTDEV: " + str(
        stdev) + " microns. \nMAX OFFSET: " + str(
            max_z) + " microns. MIN OFFSET: " + str(
                min_z) + " microns. \nPEAK TO PEAK DISTANCE: " + str(
                    ptp) + ". \nFIT PLANE: " + str(plane)

    #statement that writes the data to a file
    f = open(file, "w")
    f.write(text)
    f.close()

    #notification of completion
    print("overview written to " + file + "\nnext...")

#plotting a histogram of plane-offset values, plus a 2d scatter plot overlay of peak and failure points. 
def histogram(data, fileroot):
    file = fileroot + "/histogram.png"

    fig = Figure(figsize=(4,5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=frame_results)
    canvas.get_tk_widget().grid(column=2, row=1)

    ax=fig.add_subplot(111)
    ax.set_title('Histogram of plane offsets')
    ax.set_xlabel('Offset from best fit plane (microns)')
    ax.set_ylabel('Frequency')
    
    z_list = [i[2] * 1000 for i in data]
    bins = np.arange(min(z_list)-2, max(z_list) + 2, 2)

    ax.hist(z_list, bins=bins)

    #showing figue on GUI
    canvas.draw()

    #saving file to disk
    fig.savefig(file)

    #notification of completion
    print("histogram saved to " + file + "\nnext...")

#plotting a top-down heatmap of plane-offset values. 
def heatmap(data, ePoints, pPoints, fileroot):
    file = fileroot + "/heatmap.png"

    fig = Figure(figsize=(4,5))
    ax=fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master = frame_results)   
    canvas.get_tk_widget().grid(column=0, row=1)

    #creating a logarithmic coloring and associated colorbar
    cmap = plt.colormaps["coolwarm"]
    norm = colors.SymLogNorm(linthresh=10, linscale=0.3, vmin=-150.0, vmax=150.0, base=75)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.ax.zorder = -1

    ax.set_title("Top-Down Heatmap of plane offsets (microns)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    #setting x,y,z data lists
    x_list = data[:,0]
    y_list = data[:,1]
    z_list = [i[2] * 1000 for i in data]

    ax.set_xlim(min(x_list), max(x_list))
    ax.set_ylim(min(y_list), max(y_list))

    #creating an evenly spaced 2d grid of points betwen minimum and maximum of x and y dat
    x_coords = np.linspace(min(x_list), max(x_list))
    y_coords = np.linspace(min(y_list), max(y_list))
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)

    #creating an interpolation function to give interpolated values for each evenly spaced point in the previously created grid. 
    f = interp.LinearNDInterpolator(list(zip(x_list, y_list)), z_list)
    z_coords = f(x_coords, y_coords)
    ax.pcolormesh(x_coords, y_coords, z_coords, cmap=cmap, norm=norm)

    #scatter plots for fail and peak points
    if len(ePoints) > 0:
        scatter(ePoints, ax, color='red', label="eccentricities")

    scatter(pPoints, ax, color='orange', label="peaks")

    #this bit creates text labels notating the height of both max and min points
    strMod = ["MAX", "MIN"]
    for i in range(len(pPoints)):
        x = pPoints[i, 0] - 15
        y = pPoints[i, 1]
        offsetVal = round(pPoints[i, 2]*1000, ndigits=2)
        string = strMod[i] + ": " + str(offsetVal) + " microns"
        ax.text(x, y, string, size=7, zorder=10, color='k')

    #showing figure on GUI
    canvas.draw()

    #saving file to disk
    fig.savefig(file)

    #notification of completion
    print("heatmap saved to " + file + "\nnext...")

#plotting a 3d surface plot of plane-offset values, plus a 3d scatter plot overlay of peak and failure points. 
def residualsPlot(data, ePoints, pPoints, fileroot):
    file = fileroot + "/scatter.png"
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    frame_scatter = Frame(frame_results)
    frame_scatter.grid(column=1, row=1)
    canvas = FigureCanvasTkAgg(fig, master = frame_scatter)
    canvas.get_tk_widget().pack(side=TOP)
    
    #logarithmic coloration
    cmap = plt.colormaps["coolwarm"]
    norm = colors.SymLogNorm(linthresh=10, linscale=0.3, vmin=-150.0, vmax=150.0, base=75)

    ax.set_title("Scatter plot of distance to best fit plane")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("best-fit plane offset")

    #toolbar for mouse-exploring through the 3d plot
    toolbar = NavigationToolbar2Tk(canvas, frame_scatter)
    toolbar.update()
    toolbar.pack(side=TOP)

    #setting x,y,z values
    x_list = data[:,0]
    y_list = data[:,1]
    z_list = [i[2] * 1000 for i in data]

    ax.set_xlim(min(x_list), max(x_list))
    ax.set_ylim(min(y_list), max(y_list))
    ax.set_zlim(-150, 150)
    
    ax.plot_trisurf(x_list, y_list, z_list, cmap=cmap, norm=norm, alpha=0.75, zorder=3)

    #scatter plots of peak, fail points
    if len(ePoints) > 0:
        scatter(ePoints, ax, 1000, "red", "out of bounds", True)
    scatter(pPoints, ax, 1000, "orange", "peaks", True)

    #show figure on GUI
    canvas.draw() 

    #save file to disk
    fig.savefig(file)

    #notification of completion
    print("scatter plot saved to " + file + "\nnext...")

#scatter-plotting method called from within histogram() and residualsPlot() to plot peak, fail points. 
def scatter(points, ax, z_scale=1, color='k', label=None, spatial=False):
    #print(points)
    x_list = points[:,0]
    y_list = points[:,1]

    #the spatial argument controls if the plot is 3d or not
    if spatial:
        z_list = [i[2] * z_scale for i in points]
        ax.scatter(x_list, y_list, z_list, color=color, label=label)
    else:
        ax.scatter(x_list, y_list, color=color, label=label)

#re-initializes the GUI and returns all elements to their starting states. 
def reset():
    global file
    for widget in frame_results.winfo_children():
        if not type(widget) == Label:
            widget.destroy()
    label_file_explorer.configure(text="Local Flatness Analyzer")
    label_goodness.configure(text = "Waiting for stave core data...", bg = root.cget('bg'), fg = "black")
    button_analyze.configure(relief=RAISED, command=confirmAnalysis)
    button_explore.configure(relief=RAISED, command=browseFiles)
    button_reset.configure(relief=SUNKEN, command=NONE)
    entry_plane.delete(0, range(len(norm.get())))
    file = ""

#necessary for TKinter to work
root = Tk()

#boolean variable, controlled by checkbox, that controls whether the outliers are removed or not in analyzeFile()
zMinimum = BooleanVar(value=True)
defect = BooleanVar(value=False)
norm = StringVar()

#defining default state for all GUI elements (besides figures)
root.title("Local Flatness Analyzer")

frame_control = LabelFrame(root, text="control")
frame_analyze = LabelFrame(root, text="tools")
frame_results = LabelFrame(root, text = "results")

label_file_explorer = Label(root, text = "Local Flatness Analyzer", width = 100, height = 4, fg = "blue") 
label_space = Label(root, height=1) 
label_goodness = Label(frame_results, bd= 3, padx= 2, relief=SUNKEN, text = "Waiting for stave core data...", bg = root.cget('bg'), fg = "black", height= 2) 

button_explore = Button(frame_control, text = "Browse Files", command = browseFiles, width=10, height=1, cursor= "hand2")
button_outliers = Checkbutton(frame_analyze, text = "Enforce Z-Min", variable = zMinimum, height = 2, width = 12)
button_defect = Checkbutton(frame_analyze, text = "Defect matrix?", variable = defect, height = 2, width = 12)  
button_analyze = Button(master = root, command = confirmAnalysis, text = "Analyze", width=20, height=7, cursor= "hand2", bg="dark green", fg="light grey")
button_reset = Button(frame_control, relief=SUNKEN, command = NONE, text = "Reset", width=10, height=1, cursor= "hand2")
button_exit = Button(frame_control, text = "Exit", command = exit, width=10, height=1, cursor= "hand2")
entry_plane = Entry(frame_analyze, textvariable = norm, width = 20)
label_plane = Label(frame_analyze, text = "Norm. Vec. of Fit Plane")

#spacing for all GUI elements
weights = [2,1,1,1,1,1,2]
rootColumns = 7
for i in range(0,rootColumns):
    root.columnconfigure(weight=weights[i], index=i)
for i in range(1,6):
    frame_results.columnconfigure(weight=weights[i], index=i)

label_file_explorer.grid(column=4, row=0, sticky=N)
frame_results.grid(column=0, row=2, columnspan=rootColumns, sticky=N)
frame_control.grid(column=2, row=1, sticky=N)
frame_analyze.grid(column=4, row=1, sticky=N)
button_analyze.grid(column=5,row=1, sticky=NW)

button_outliers.grid(column=0, row=1, sticky=W)
button_defect.grid(column=0, row=0, sticky=W)
label_plane.grid(column=1, row=0)
entry_plane.grid(column=1, row=1, padx=[10,10])
button_explore.pack(side=TOP)
button_reset.pack(side=TOP)
button_exit.pack(side=TOP)
label_goodness.grid(row=0, column=1)

label_goodness.update()
root.mainloop()