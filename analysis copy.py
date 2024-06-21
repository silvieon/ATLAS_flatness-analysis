import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import nums
import scipy.interpolate as interp
from tkinter import *
from tkinter import filedialog
from skspatial.objects import Points, Plane
#from skspatial.plotting import plot_3d
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm, colors

global file
file = ""
global dirmemory
dirmemory = "/"

#opens and reads requested data source file
def browseFiles():
    global file 
    global dirmemory
    file = filedialog.askopenfilename(initialdir = dirmemory, title = "Select a File", filetypes = (("Text files","*.txt*"),("all files","*.*")))
    filerev = file[::-1]
    print(filerev)
    index = filerev.find("/")
    filerev = filerev[index:len(filerev)]
    dirmemory = filerev[::-1]
    print("Open file explorer at: " + dirmemory)
    label_file_explorer.configure(text="File Opened: " + file)

def confirmAnalysis():
    global file 
    plaintext = ""
    if file == "":
        label_file_explorer.configure(text="You have not opened a file! Please select a file.")
    else:
        plaintext = readFile(file)
    
    if plaintext.find("Plane   ") == -1: 
        analyzeFile(plaintext, file)
        file = ""
    
    else:
        if not os.path.isdir(file[0:len(file)-4]):
            os.makedirs(file[0:len(file)-4])
        try:
            splitFile(plaintext, file)
        except:
            print("error occurred with file: check file contents.")


def readFile(file):
    button_analyze.configure(relief=SUNKEN, command=NONE)
    button_explore.configure(relief=SUNKEN, command=NONE)
    button_reset.configure(relief=RAISED, command=reset)

    plaintext = ""

    with open(file, "r") as text:
        plaintext = text.read()
    
    return plaintext

def splitFile(text, file):
    numberindex = file.find("-") - 2
    fileNumber = 0
    if not file[numberindex] == "1":
        fileNumber = file[numberindex + 1]
    else:
        fileNumber = int(file[numberindex:numberindex+2])
    index = text.find("Plane   ")

    while index > -1:
        string = text[0:index]
        f = open(file[0:len(file)-4] + "/moduleLocation" + str(fileNumber) + ".txt", "w")
        f.write(string)
        f.close()
        fileNumber = fileNumber + 1
        temp = text[index:len(text)]
        index2 = temp.find("Point   ")
        text = temp[index2:len(temp)]
        index = text.find("Plane   ")
        print("writing to " +file[0:len(file)-4] + "/moduleLocation" + str(fileNumber) + ".txt" + "...")

    print("done")

    label_file_explorer.configure(text="Multi-module data file split into single-module data files.")

def analyzeFile(plaintext, file):
    print("analyzing, doo de doo dooo~")
    outlier = outlierSense.get()

    data = reformatString(plaintext)

    if outlier:
        data = removeOutliers(data)

    data = data.to_numpy(dtype=float)
    p = Points(data)
    plane = Plane.best_fit(p)

    peakDistances = [0,0]
    peakIndices = [0,0]
    peakPoints = [data[0], data[1]]

    eccentricPoints = []
    eccentricIndices = []
    eccentricDistances = []

    planeDistances = []

    for i in range(len(p)): 
        point = data[i]
        planeDistance = plane.distance_point_signed(point)
        if abs(planeDistance) >= 75/1000:
            eccentricPoints.append(data[i])
            eccentricIndices.append(i)
            eccentricDistances.append(planeDistance)
        if planeDistance >= peakDistances[0]:
            peakDistances[0] = planeDistance
            peakPoints[0] = data[i]
            peakIndices[0] = i
        if planeDistance <= peakDistances[1]: 
            peakDistances[1] = planeDistance
            peakPoints[1] = data[i]
            peakIndices[1] = i
        planeDistances.append(planeDistance)

    #removeIndices = eccentricIndices + peakIndices
    #normalPoints= np.delete(numpy, removeIndices, axis=0)

    ePoints = np.array(eccentricPoints, dtype=float)
    pPoints = np.array(peakPoints, dtype=float)

    if not os.path.isdir(file[0:len(file)-4]):
            os.makedirs(file[0:len(file)-4])

    histogram(planeDistances)

    heatmap(ePoints, eccentricDistances, pPoints, peakDistances, data, planeDistances)

    residualsPlot(data, planeDistances)

    labelConfigure(eccentricDistances)

    label_file_explorer.configure(text="Saved figures to folder: "+ file[0:len(file)-4])

    fileroot = file[0:len(file)-4]

    to_CSV(data, planeDistances, fileroot)

    text_label(planeDistances, peakDistances, fileroot)

def removeOutliers(dataframe):
    print("removing outliers...")
    z_vals = np.array(dataframe.iloc[:,2], dtype=float)
    removeindices = []
    z_med = np.median(z_vals)
    for i in range(len(z_vals)):
        diff = z_med - z_vals[i]
        if diff > 0.25:
            removeindices.append(i)
            print("removed! z-difference: " + str(diff))
    if len(removeindices) == 0:
        print("woah! no outliers to remove. moving on...")
    return dataframe.drop(index=removeindices)

#turns space-separated proprietary data file format into a Pandas dataframe
def reformatString(text):

    plus = "+"
    minus = "-"

    #SEARCHES FOR + OR -
    res = [i for i in range(len(text)) if text.startswith(plus, i) or text.startswith(minus, i)]
    nums = [text[i:i+10] for i in res]
    checker = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "."]
    for i in nums:
        truth = FALSE
        if not i[4:5] == ".":
            truth = TRUE
        for j in range(len(i)-1):
            ch = i[j:j+1]
            if not ch in checker:
                truth = TRUE
        if truth:
            nums.remove(i)

    numsMod0 = []
    numsMod1 = []
    numsMod2 = []

    for i in range(len(nums)):
        match i%3:
            case 0:
                numsMod0.append(nums[i])
            case 1:
                numsMod1.append(nums[i])
            case 2:
                numsMod2.append(nums[i])

    removeIndices = []
    
    for j in range(len(numsMod1)):
        y = float(numsMod1[j])
        if abs(y) < 0.001:
            removeIndices.append(j)
            print("removed a bad point -- plane location out of range")

    for index in sorted(removeIndices, reverse=True):
        del numsMod0[index]
        del numsMod1[index]
        del numsMod2[index]

    if not len(numsMod0) == len(numsMod1) and len(numsMod1) == len(numsMod2):
        print("list 1 length: " + str(len(numsMod0)))
        print("list 2 length: " + str(len(numsMod1)))
        print("list 3 length: " + str(len(numsMod2)))
        label_file_explorer.configure(text="Oh no! We've run into an error. Check data file formatting.")
        return None
    
    dict = {'X/R Location': numsMod0, 'Y/A Location': numsMod1, 'Z Location': numsMod2}

    df = pd.DataFrame(dict)
    
    return df

def to_CSV(numpy, planeDistances, fileroot):
    distances = []
    for i in planeDistances:
        distances.append(i)
    file = fileroot + "/offsets.csv"
    temp = -1000
    index = 0
    nums = []
    array = []
    row = 0
    while index < len(distances):
        x_val = numpy[index, 0]
        z_val = np.round(distances[index], 6)
        if x_val <= temp:
            row = row + 1
            array.append(nums)
            nums = []
        nums.append(z_val)
        temp = x_val
        index = index + 1
    array.append(nums)
    
    csv = pd.DataFrame(array)
    csv.to_csv(file)

    print("CSV data holds " + str(len(distances)) + " points. ")

    print("CSV data saved to " + file)

def text_label(planeDistances, peakDistances, fileroot):
    file = fileroot + "/overview.txt"
    stdev = np.std(planeDistances)
    stdev = stdev * 1000
    max = peakDistances[0] * 1000
    min = peakDistances[1] * 1000
    text = "OVERVIEW. \nSTDEV: " + str(stdev) + " microns. MAX OFFSET: " + str(max) + " microns. MIN OFFSET: " + str(min) + " microns. "

    f = open(file, "w")
    f.write(text)
    f.close()

    print("text overview written to " + file)

def labelConfigure(eccentricDistances):
    textOut = "Waiting for stave core data..."
    bgColor = root.cget('bg')
    fgColor = "black"

    cap = FALSE
    dip = FALSE

    for i in eccentricDistances:
        if i < 0:
            dip = TRUE
    
    for i in eccentricDistances:
        if i > 0:
            cap = TRUE

    truth = int(cap) + int(dip)*2
    #0: neither cap nor dip, 1: cap, 2: dip, 3: both

    match truth:
        case 3:
            textOut = "STAVE CORE FAILED."
            bgColor = "orange red"
            fgColor = "black"
            #this corresponds to the rejection notice
        case 2:
            textOut = "B-CLASS STAVE CORE."
            bgColor = "gold"
            fgColor = "black"
            #this corresponds to the B-class stave core notice
        case 1:
            textOut = "STAVE CORE FAILED."
            bgColor = "orange red"
            fgColor = "black"
            #this corresponds to the rejection notice
        case 0: 
            textOut = "STAVE CORE PASSED."
            bgColor = "dark green"
            fgColor = "white"
            #this corresonds to the good stave core notice

    label_goodness.configure(text=textOut, bg = bgColor, fg = fgColor)

def histogram(planeDistances):
    global file
    array = []

    fig = Figure(figsize=(4,5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=RIGHT, expand= "y")

    for i in range(len(planeDistances)):
        array.append(planeDistances[i]*1000)

    p = fig.gca()
    p.hist(array, bins=np.arange(min(array), max(array) + 2, 2))
    p.set_title('Histogram of plane offsets')
    p.set_xlabel('Offset from best fit plane (microns)')
    p.set_ylabel('Frequency')
    canvas.draw()

    savepath = file[0:len(file)-4] + "/histogram.png"
    fig.savefig(savepath)

    print("histogram saved to " + savepath)

def heatmap(ePoints, eDistances, pPoints, pDistances, numpy, planeDistances):
    global file

    fig = Figure(figsize=(4,5))
    ax=fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master = root)   
    canvas.get_tk_widget().pack(side=LEFT, expand= "y")
    savepath = file[0:len(file)-4] + "/heatmap.png"

    cmap = plt.colormaps["coolwarm"]
    norm = colors.SymLogNorm(linthresh=10, linscale=0.3, vmin=-150.0, vmax=150.0, base=75)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.ax.zorder = -1

    ax.set_title("Top-Down Heatmap (microns)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    x_list = numpy[:,0]
    y_list = numpy[:,1]
    z_list = []

    for i in range(len(planeDistances)):
        z_list.append(planeDistances[i]*1000)

    ax.set_xlim(min(x_list), max(x_list))
    ax.set_ylim(min(y_list), max(y_list))

    x_coords = np.linspace(min(x_list), max(x_list))
    y_coords = np.linspace(min(y_list), max(y_list))
    x_coords, y_coords=np.meshgrid(x_coords, y_coords)

    f=interp.LinearNDInterpolator(list(zip(x_list, y_list)), z_list)
    z_coords = f(x_coords, y_coords)
    ax.pcolormesh(x_coords, y_coords, z_coords, cmap=cmap, norm=norm)

    if len(eDistances) > 0:
        ax.scatter(ePoints[:,0], ePoints[:,1], color='red', label="eccentric points")

    ax.scatter(pPoints[:,0], pPoints[:,1], color= 'yellow', label="peaks")

    strMod = ["MAX", "MIN"]
    for i in range(len(pPoints)):
        x = pPoints[i, 0] - 15
        y = pPoints[i, 1]
        offsetVal = round(pDistances[i]*1000, ndigits=1)
        string = strMod[i] + ": " + str(offsetVal) + " microns"
        ax.text(x, y, string, size=7, zorder=10, color='k')

    canvas.draw()

    fig.savefig(savepath)

    print("heatmap saved to " + savepath)

def residualsPlot(numpy, planeDistances):
    global file  
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master = root)

    ax.set_title("Scatter Plot w/ Fit Plane")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    x_list = numpy[:,0]
    y_list = numpy[:,1]
    z_list = []

    for i in range(len(planeDistances)):
        z_list.append(planeDistances[i]*1000)

    ax.set_xlim(min(x_list), max(x_list))
    ax.set_ylim(min(y_list), max(y_list))
    ax.set_zlim(-75, 75)

    cmap = plt.colormaps["coolwarm"]
    norm = colors.SymLogNorm(linthresh=10, linscale=0.3, vmin=-150.0, vmax=150.0, base=75)
    ax.plot_trisurf(x_list, y_list, z_list, cmap=cmap, norm=norm, alpha=0.75, zorder=3)

    canvas.draw() 

    canvas.get_tk_widget().pack(side=TOP, expand= False)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack(side=TOP, expand= False)

    savepath = file[0:len(file)-4] + "/scatter.png"
    fig.savefig(savepath)

    print("scatter plot saved to " + savepath)

def scatterPlot(eccentricPoints, peakPoints, normalPoints, plane, peakDistances, numpy): 
    global file  
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master = root) 

    ax.set_title("Scatter Plot w/ Fit Plane")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    x_list = numpy[:,0]
    y_list = numpy[:,1]

    ax.set_xlim(min(x_list), max(x_list))
    ax.set_ylim(min(y_list), max(y_list))
    ptp_x = np.ptp(x_list)
    ptp_y = np.ptp(y_list)
    plane_lims_x = [-0.5*ptp_x, 0.5*ptp_x]
    plane_lims_y = [-0.5*ptp_y, 0.5*ptp_y]

    z_mid = np.median(numpy[:,2])
    ax.set_zlim(z_mid - 0.075, z_mid + 0.075)

    cmap = plt.colormaps["twilight_shifted"]

    ax.plot_trisurf(x_list, y_list, numpy[:,2], cmap=cmap, vmin=-0.075, vmax=0.075, alpha=0.75, zorder=3)

    if len(eccentricPoints) > 0:
        ePoints = Points(eccentricPoints)
        ePoints.plot_3d(ax, s=20, c = 'red', zorder=4)

    pPoints = Points(peakPoints)
    pPoints.plot_3d(ax, s=20, c = 'orange', zorder=5)

    #strMod = ["MAX.", "MIN."]
    #for i in range(len(peakPoints)):
    #    x = peakPoints[i][0]
    #    y = peakPoints[i][1]
    #    z = peakPoints[i][2]
    #    offsetVal = round(peakDistances[i], ndigits=4)
    #    string = strMod[i] + "perp. offset: " + str(offsetVal)
    #    ax.text(x,y,z, string, size=7, zorder=0, color='k')

    #nPoints = Points(normalPoints)
    #nPoints.plot_3d(ax, s =10, c= 'blue')
    
    plane.plot_3d(ax, lims_x=plane_lims_x, lims_y=plane_lims_y, alpha=0.5)
    plane.point.plot_3d(ax, s=100)
  
    canvas.draw() 

    canvas.get_tk_widget().pack(side=TOP, expand= False)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack(side=TOP, expand= False)

    savepath = file[0:len(file)-4] + "/scatter.png"
    fig.savefig(savepath)

    print("scatter plot saved to " + savepath)

def reset():
    global file
    for widget in root.winfo_children():
        if isinstance(widget, Canvas):
            widget.destroy()
        if isinstance(widget, NavigationToolbar2Tk):
            widget.destroy()
    label_file_explorer.configure(text="Local Flatness Analyzer")
    label_goodness.configure(text = "Waiting for stave core data...", bg = root.cget('bg'), fg = "black")
    button_analyze.configure(relief=RAISED, command=confirmAnalysis)
    button_explore.configure(relief=RAISED, command=browseFiles)
    button_reset.configure(relief=SUNKEN, command=NONE)
    file = ""

root = Tk()

outlierSense = BooleanVar(value=True)

root.title("Local Flatness Analyzer")
label_file_explorer = Label(root, text = "Local Flatness Analyzer", width = 100, height = 4, fg = "blue") 
label_space = Label(root, height=1) 
label_goodness = Label(root, bd= 3, padx= 2, relief=SUNKEN, text = "Waiting for stave core data...", bg = root.cget('bg'), fg = "black", height= 2) 

button_explore = Button(root, text = "Browse Files", command = browseFiles, width=10, height=1, cursor= "hand2")
button_outliers = Checkbutton(root, text = "Remove Outliers", variable = outlierSense, height = 2, width = 12) 
button_analyze = Button(master = root, command = confirmAnalysis, text = "Analyze", width=10, height=1, cursor= "hand2")
button_reset = Button(master = root, relief=SUNKEN, command = NONE, text = "New Analysis", width=10, height=1, cursor= "hand2")
button_exit = Button(root, text = "Exit", command = exit, width=10, height=1, cursor= "hand2")
 

label_file_explorer.pack(side=TOP)
button_explore.pack(side=TOP)
button_outliers.pack(side=TOP)
button_analyze.pack(side=TOP)
button_reset.pack(side=TOP)
button_exit.pack(side=TOP)
label_space.pack(side=TOP)
label_goodness.pack(side=TOP)

label_goodness.update()
root.mainloop()