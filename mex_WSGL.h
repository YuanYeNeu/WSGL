/*
 *This program is modified based on the Compact Watershed codes, by YUAN Ye, yuanye_neu@163.com. 
 *If you use these codes, please cite the correspoding paper: Watershed-based Superpixels with Global and Local Boundary Marching
 *
 *This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 *This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *We would like to thank Peer Neubert et al. for their work. 
 *The original statement is shown as below.
 */
/*
 * Compact Watershed
 * Copyright (C) 2014  Peer Neubert, peer.neubert@etit.tu-chemnitz.de
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */ 

#ifndef MEX_WSGL_H
#define MEX_WSGL_H

#include "opencv2/opencv.hpp"
#include <stdio.h>


/**
Compact watershed is a superpixel or oversegmentation algorithm.
For details have a look at:

"Compact Watershed and Preemptive SLIC:\\On improving trade-offs of superpixel segmentation algorithms"
Peer Neubert and Peter Protzel, ICPR 2014

*/
void compact_watershed(cv::Mat& img, cv::Mat& B, float compValStep, cv::Mat& seeds,int spn, int postprocessing,int ItrSet, int DeltaC, int seedsType,int * labelnumber);

#endif