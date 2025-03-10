import cv2
import numpy as np
import rhinoscriptsyntax as rs
import csv

def show_rhino_screen():
    rs.ZoomExtents()
    rs.Redraw()

image_path = "/mnt/data/file-F9QDn9bp1LSfSqd4ytU9tj"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

curves = []
for cnt in contours:
    points = [tuple(pt[0]) + (0,) for pt in cnt]      
    curve = rs.AddInterpCurve(points)
    curves.append(curve)

extruded_objects = []
for curve in curves:
    if curve:
        extruded = rs.ExtrudeCurveStraight(curve, (0, 0, 0), (0, 0, 10))
        extruded_objects.append(extruded)

csv_file_path = "/mnt/data/points.csv"

try:
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])  
            rs.AddPoint(x, y, z)
            print("Point added: ({}, {}, {})".format(x, y, z))

except Exception as e:
    print("CSV file error: {}".format(e))  

output_stl_path = "/mnt/data/output_model.stl"

try:
    rs.Command('-Export "{}" _Enter'.format(output_stl_path)) 
    print("STL file saved to:", output_stl_path)
except Exception as e:
    print("STL export error: {}".format(e))

show_rhino_screen()
