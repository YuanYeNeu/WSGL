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

# include "mex_helper_WSGL.h"
# include <time.h>
using namespace std;
using namespace cv;


//=============================================================================
int convertMx2Mat(mxArray const *mx_in, Mat& mat_out)
{
      
  //figure out dimensions  
  int h, w, c;
  const mwSize *dims = mxGetDimensions(mx_in);
  int numdims = mxGetNumberOfDimensions(mx_in);
  if(numdims==2)
  {
    h = (int)dims[0]; 
    w = (int)dims[1];
    c = 1;
  }
  else if(numdims==3)
  {
    h = (int)dims[0]; 
    w = (int)dims[1];
    c = (int)dims[2];
  }
  else
  {
    cerr << "Unsupported number of dimensions in convertMx2Mat(): "<<numdims<<endl;
    return -1;
  }

  
  // check number of channels
  if(numdims==3 && c!=3)
  {
    cerr << "Unsupported number of color channels in convertMx2Mat(): "<< c <<endl;
    return -1;
  }
  
  // figure out type
  int type = mxGetClassID(mx_in);

  if(type == mxUINT8_CLASS)
    mat_out = (c==1) ? cv::Mat(w, h, CV_8UC1) : cv::Mat(w, h, CV_8UC3);
  else if(type == mxINT8_CLASS)
    mat_out = (c==1) ? cv::Mat(w, h, CV_8SC1) : cv::Mat(w, h, CV_8SC3);
  else if(type == mxINT32_CLASS)
    mat_out = (c==1) ? cv::Mat(w, h, CV_32SC1) : cv::Mat(w, h, CV_32SC3);
  else if(type == mxSINGLE_CLASS)
    mat_out = (c==1) ? cv::Mat(w, h, CV_32FC1) : cv::Mat(w, h, CV_32FC3);
  else
  {
    cerr << "unsupported type in convertMx2Mat: "<< (int)type <<endl;
    cerr << "supported are :"<<endl;
    cerr << "   mxUINT8_CLASS="<<(int)mxUINT8_CLASS<<endl;
    cerr << "   mxINT8_CLASS="<<(int)mxINT8_CLASS<<endl;
    cerr << "   mxINT32_CLASS="<<(int)mxINT32_CLASS<<endl;
    cerr << "   mxSINGLE_CmxSINGLE_CLASSLASS="<<(int)mxSINGLE_CLASS<<endl;
    return -1;
  }
  
  // handle one channel images
  if( c==1 )
  {
      // copy data
      int stepWidth = mat_out.step; // number of bytes per row (this already takes care of elemSize)
      int nBytes = mat_out.elemSize(); // number of bytes for each element

      char* mx_in_data = (char*)mxGetData( mx_in );
      char* data = (char*)mat_out.data;
      if(stepWidth != w*nBytes)
      {
        // we have to copy (Matlab) column to OpenCV row
        for(int i=0; i<w; i++) // <-- iterate over Matlab columns (-->w) and OpenCV rows
        {
          memcpy(&data[i*stepWidth], &mx_in_data[i*h*nBytes], h*nBytes);
        }    
      }
      else
      {
        // we can copy as one block
        memcpy(data, mx_in_data, w*h*nBytes);
      }
  }
  else if(c==3 && type==mxUINT8_CLASS)
  {
     
      // get pointer to input data and fill buffer
      unsigned char* I = (unsigned char*) mxGetPr(mx_in);    
      unsigned int R, G, B;
      unsigned int idx;
      unsigned int n=h*w;
      unsigned int nn=n+n;
      for(int j=0; j<w; j++)
      {
        for(int i=0; i<h; i++)
        {
          // read in column major order from Matlab and write row major in OpenCV
          // i: Matlab y, OpenCV x
          // j: Matlab x, OpenCV y
          idx = (h*j)+i; // Matlab Index
          R = I[idx];
          G = I[idx + n];
          B = I[idx + nn];

          Vec3b pixel_color(B, G, R);
          mat_out.at<Vec3b>(j,i) = pixel_color;  
        }
      }
  }
  else
  {
    cerr << "Problem in convertMx2Mat"<<endl;
    return -1;
  }
  
  return 0;
  
}


//=============================================================================
int convertMat2Mx(cv::Mat const &mat_in, mxArray*& mx_out)
{
  
  int h = mat_in.rows;
  int w = mat_in.cols;  
  int type = mat_in.type();
  
  // dimension of Matlab array
  int dims[2]; 
  dims[0] = w; 
  dims[1] = h; 
    
  // create Matlab matrix
  if(type == CV_8UC1)
    mx_out = mxCreateNumericArray(2, dims, mxUINT8_CLASS, mxREAL);  
  else if(type == CV_8SC1)
    mx_out = mxCreateNumericArray(2, dims, mxINT8_CLASS, mxREAL);  
  else if(type == CV_32SC1)
    mx_out = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);  
  else if(type == CV_32FC1)
    mx_out = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  else
  {
    cerr << "unsupported type convertMat2Mx"<<endl;
    return -1;
  }
      
  // copy data
  int stepWidth = mat_in.step; // number of bytes per row (this already takes care of elemSize)
  int nBytes = mat_in.elemSize(); // number of bytes for each element
  char* mx_out_data = (char*)mxGetData( mx_out );
  char* data = (char*)mat_in.data;
  if(stepWidth != w*nBytes)
  {
    // we have to copy row per row
   for(int i=0; i<h; i++)
    {
      memcpy(&mx_out_data[i*w*nBytes], &data[i*stepWidth], w*nBytes);
    }    
  }
  else
  {
    // we can copy as one block
    memcpy(mx_out_data, data, w*h*nBytes);
  }
  
  return 0;
}

