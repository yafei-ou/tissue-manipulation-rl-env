// Gmsh project created on Wed Aug 03 09:20:15 2022
SetFactory("OpenCASCADE");
//+
Point(1) = {-10, -10, 0, 1.0};
//+
Point(2) = {-10, 10, 0, 1.0};
//+
Point(3) = {10, 10, 0, 1.0};
//+
Point(4) = {10, -10, 0, 1.0};
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {1, 2};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
MeshSize {2, 1, 4, 3} = 0.5;
