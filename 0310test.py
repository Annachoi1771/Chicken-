import cv2
import numpy as np
import rhinoscriptsyntax as rs

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

output_stl_path = "/mnt/data/output_model.stl"
rs.Command(f'-Export "{output_stl_path}" _Enter')


show_rhino_screen()

