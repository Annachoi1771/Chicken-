import csv
import rhinoscriptsyntax as rs

filename = "C:\Users\clearpc\Downloads\chicken_shape_new.csv"



with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            z = float(0)
            print x, y, z
            rs.AddPoint(x,y,z)