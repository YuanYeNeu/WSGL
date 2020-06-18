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
#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include<iostream>

#include "mex_helper_WSGL.h"
#include "mex_WSGL.h"

using namespace std;
using namespace cv;

/* compile with
 *  mexOpenCV mex_WSGL.cpp mex_helper_WSGL.cpp
 */


void compact_watershed(Mat& img, Mat& B, float compValStep, Mat& seeds, int n, int postprocessing,int ItrSet, int DeltaC, int seedsType, int * labelnumber)
{
    
    Mat markers = Mat::zeros(img.rows, img.cols, CV_32SC1);
    int labelIdx = 0;
    int disAver,MQ;
    if (seedsType==0)
    {
        if (seeds.empty())
        {
            //distribute initial markers
            float ny = sqrt((float(n)*img.rows) / img.cols);
            float nx = float(n) / ny;
            
            float dx = img.cols / nx;
            float dy = img.rows / ny;
            
            if (dx>dy)
            {
                disAver=ceil(dx/2);
            }
            else
            {
                disAver=ceil(dy/2);
            }
            if (disAver<10)
                MQ=10;
            else
                MQ=disAver;
            
            for (float i = dy / 2; i<markers.rows; i += dy)
            {
                for (float j = dx / 2; j<markers.cols; j += dx)
                {
                    labelIdx++;
                    markers.at<int>(floor(i), floor(j)) = labelIdx;
                }
            }
        }
        else
        {
            markers=seeds;
        }
    }
    else
    {
        if (seeds.empty())
        {
            // distribute initial markers
            float ny = sqrt((float(n)*img.rows) / img.cols);
            float nx = float(n) / ny;
            
            float dx = img.cols / nx;
            float dy = img.rows / ny;
            
            if (dx>dy)
            {
                disAver=ceil(dx/2);
            }
            else
            {
                disAver=ceil(dy/2);
            }
            if (disAver<10)
                MQ=10;
            else
                MQ=disAver;
            int sord=0;
            for (float i = dy / 2; i<markers.rows; i += dy)
            {
                sord++;
                for (float j = dx / 2; j+dx/4<markers.cols; j += dx)
                {
                    labelIdx++;
                    if (sord%2==0)
                        markers.at<int>(floor(i), floor(j-dx/4)) = labelIdx;
                    else
                        markers.at<int>(floor(i), floor(j+dx/4)) = labelIdx;
                }
            }
        }
        else
        {
            markers=seeds;
        }
    }
    
    
    
    cvWatershed( img, markers, compValStep,labelIdx,n,postprocessing,disAver,ItrSet,MQ, DeltaC,labelnumber);
    
    B=markers;
    
}

typedef struct CvWSNode
{
    struct CvWSNode* next;
    int mask_ofs;
    int img_ofs;
    float compVal;
}
CvWSNode;

typedef struct CvWSQueue
{
    CvWSNode* first;
    CvWSNode* last;
}
CvWSQueue;

static CvWSNode* icvAllocWSNodes(CvMemStorage* storage)
{
    CvWSNode* n = 0;
    
    int i, count = (storage->block_size - sizeof(CvMemBlock)) / sizeof(*n) - 1;
    
    n = (CvWSNode*)cvMemStorageAlloc(storage, count*sizeof(*n));
    for (i = 0; i < count - 1; i++)
        n[i].next = n + i + 1;
    n[count - 1].next = 0;
    
    return n;
}


void  cvWatershed(Mat& imgRGB, Mat& BMat,  float compValStep, int labelId,int spn, int postProcessing,int disAver,int itrSet,int MQs, int deltaC, int* labelnumber)
{
    
    Mat imgMat;
    cvtColor(imgRGB, imgMat, COLOR_BGR2HSV);
    
    const int IN_QUEUE = -2;
    const int WSHED = -1;
    const int RELABELED=-4;
    const int NQ =  2048;
    const int MQ=MQs;
    
    cv::Ptr<CvMemStorage> storage;
    Mat labelMat=Mat::zeros(labelId, 3, CV_32SC1);
    CvMat sstub;
    CvMat dstub;
    CvSize size;
    CvWSNode* free_node = 0, *node;
    CvWSQueue q[NQ+1];
    int active_queue;
    int i, j;
    double db, dg, dr;
    int* mask;
    int* iflabeled;
    uchar* img;
    int* labelLab;
    int* numberLab;
    int mstep, istep,labstep;
    int subs_tab[2 * NQ + 1];
    
    
    
    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])
    
#define ws_push(idx,mofs,iofs,cV)  \
    {                               \
                                            if (!free_node)            \
                                            free_node = icvAllocWSNodes(storage); \
                                            node = free_node;           \
                                                    free_node = free_node->next; \
                                                    node->next = 0;             \
                                                            node->mask_ofs = mofs;      \
                                                            node->img_ofs = iofs;       \
                                                                    node->compVal = cV;    \
                                                                    if (q[idx].last)           \
                                                                            q[idx].last->next = node; \
                                                                    else                        \
                                                                            q[idx].first = node;    \
                                                                            q[idx].last = node;         \
    }
    
#define ws_pop(idx,mofs,iofs,cV)   \
    {                               \
                                            node = q[idx].first;        \
                                            q[idx].first = node->next;  \
                                                    if (!node->next)           \
                                                    q[idx].last = 0;        \
                                                    node->next = free_node;     \
                                                            free_node = node;           \
                                                            mofs = node->mask_ofs;      \
                                                                    iofs = node->img_ofs;       \
                                                                    cV = node->compVal;       \
    }
    
    
#define c_diff(ptr1,ptr2,diff)      \
    {                                   \
                                                db = abs((ptr1)[0] - (ptr2)[0]); \
                                                dg = abs((ptr1)[1] - (ptr2)[1]); \
                                                        dr = abs((ptr1)[2] - (ptr2)[2]); \
                                                        diff = sqrt(db*db+dg*dg+dr*dr); \
    }
//
//
#define cc_diff(ptr0,lTemp,aTemp,bTemp,t)      \
    {                                   \
                                                db = double((ptr0)[0])-double(lTemp); \
                                                dg = double((ptr0)[1])-double(aTemp); \
                                                        dr =double((ptr0)[2])-double(bTemp); \
                                                        t = double(sqrt(db*db+dg*dg+dr*dr)); \
    }
    
    
    int imgrow=imgMat.rows;
    int imgcol=imgMat.cols;
    
    Mat numberMat=Mat::zeros(labelId, 1, CV_32SC1);
    Mat iflabelMat=Mat::zeros(imgrow,imgcol,CV_32SC1);
    CvMat *src = cvCreateMat(imgrow,imgcol,CV_8UC3);
    CvMat *dst = cvCreateMat(imgrow,imgcol,CV_32SC1);
    CvMat *labellabptr = cvCreateMat(labelId,3,CV_32SC1);
    CvMat *numberlabptr = cvCreateMat(labelId,1,CV_32SC1);
    CvMat *iflabel = cvCreateMat(imgrow,imgcol,CV_32SC1);
    
    
    
    CvMat temp = imgMat; //转化为CvMat类型，而不是复制数据
    cvCopy(& temp, src);
    CvMat temp0 = BMat;
    cvCopy(& temp0, dst);
    temp0=iflabelMat;
    cvCopy(& temp0, iflabel);
    CvMat temp000=labelMat;
    cvCopy(& temp000, labellabptr);
    CvMat temp0000=numberMat;
    cvCopy(& temp0000, numberlabptr);
    if (CV_MAT_TYPE(src->type) != CV_8UC3)
        CV_Error(CV_StsUnsupportedFormat, "Only 8-bit, 3-channel input images are supported");
    
    if (CV_MAT_TYPE(dst->type) != CV_32SC1)
        CV_Error(CV_StsUnsupportedFormat,
                "Only 32-bit, 1-channel output images are supported");
    
    if (!CV_ARE_SIZES_EQ(src, dst))
        CV_Error(CV_StsUnmatchedSizes, "The input and output images must have the same size");
    
    size = cvGetMatSize(src);
    storage = cvCreateMemStorage();
    
    istep = src->step;
    img = src->data.ptr;
    mstep = dst->step / sizeof(mask[0]);
    mask = dst->data.i;
    
    iflabeled=iflabel->data.i;
    labstep=labellabptr->step/sizeof(labelLab[0]);
    labelLab=labellabptr->data.i;
    numberLab=numberlabptr->data.i;
    
    memset(q, 0, (NQ+1)*sizeof(q[0]));
    
    for (i = 0; i < NQ; i++)
        subs_tab[i] = 0;
    for (i = NQ; i <= 2 * NQ; i++)
        subs_tab[i] = i - NQ;
    
    
    
    for(i=0;i<size.height;i++)
    {
        for(j=0;j < size.width; j++)
        {
            iflabeled[i*size.width+j]=0;
            if(mask[i*size.width+j]>0)
            {
                int tempmask=mask[i*size.width+j];
                labelLab[tempmask*labstep-3]=img[3*i*size.width+3*j];
                labelLab[tempmask*labstep-2]=img[3*i*size.width+3*j+1];
                labelLab[tempmask*labstep-1]=img[3*i*size.width+3*j+2];
                numberLab[tempmask-1]++;
            }
        }
    }
    
    
    for (j = 0; j < size.width; j++)
    {
        mask[j] = mask[j + mstep*(size.height - 1)] = WSHED;
        iflabeled[j] = iflabeled[j + mstep*(size.height - 1)] = RELABELED;
    }
    for (i = 1; i < size.height - 1; i++)
    {
        img += istep; mask += mstep;
        mask[0] = mask[size.width - 1] = WSHED;
        
        iflabeled[0] = iflabeled[size.width - 1] = RELABELED;
        for (j = 1; j < size.width - 1; j++)
        {
            int* m = mask + j;
            if (m[0] < 0) m[0] = 0;
            if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
            {
                uchar* ptr = img + j * 3;
                int idx=2000, t;
                
                if (m[-1] > 0)
                {
                    c_diff(ptr, ptr - 3, idx);
                }
                if (m[1] > 0)
                {
                    c_diff(ptr, ptr + 3, t);
                    idx = ws_min(idx, t);
                }
                if (m[-mstep] > 0)
                {
                    c_diff(ptr, ptr - istep, t);
                    idx = ws_min(idx, t);
                }
                if (m[mstep] > 0)
                {
                    c_diff(ptr, ptr + istep, t);
                    idx = ws_min(idx, t);
                }
                assert(0 <= idx && idx <= NQ - 1);
                if (idx>NQ-1)
                    idx=NQ-1;
                ws_push(idx, i*mstep + j, i*istep + j * 3, 0.0);
                m[0] = IN_QUEUE;
            }
        }
    }
    // find the first non-empty queue
    for (i = 0; i < NQ; i++)
        
        if (q[i].first)
            break;
    
    // if there is no markers, exit immediately
    if (i > NQ)
        return;
    
    active_queue = i;
    img = src->data.ptr;
    mask = dst->data.i;
    // recursively fill the basins
    
    for (;;)
    {
        int  aveSize=imgcol*imgrow/spn;
        int mofs, iofs;
        int lab = 0, t,diff;
        int* m;
        //int* l;
        uchar* ptr;
        
        // search for next queue
        if (q[active_queue].first == 0)
        {
            for (i = active_queue + 1; i < NQ; i++)
                if (q[i].first)
                    break;
            if (i > NQ-1)
                break;
            active_queue = i;
        }
        
        // get next element of this queue
        
        float compVal,compDiff;
        ws_pop(active_queue, mofs, iofs, compVal);
        
        m = mask + mofs; // pointer to element in mask
        ptr = img + iofs; // pointer to element in image
        
        
        // have a look at all neighbors, if they have different label, mark
        // as watershed and continue
        t = m[-1];
        if (t > 0) lab = t;
        t = m[1];
        if (t > 0)
        {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        t = m[-mstep];
        if (t > 0)
        {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        t = m[mstep];
        if (t > 0)
        {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        assert(lab != 0);
        m[0] = lab;
        if (lab == WSHED)
        {
            
            continue;
        }
        int tempmask=m[0];
        
        int lTemp;
        int aTemp;
        int bTemp;
        labelLab[tempmask*labstep-3]+=ptr[0];
        labelLab[tempmask*labstep-2]+=ptr[1];
        labelLab[tempmask*labstep-1]+=ptr[2];
        
        int valueTemp;
        numberLab[tempmask-1]+=1;
        valueTemp=numberLab[tempmask-1];
        aTemp=labelLab[tempmask*labstep-2]/valueTemp;
        bTemp=labelLab[tempmask*labstep-1]/valueTemp;
        lTemp=labelLab[tempmask*labstep-3]/valueTemp;
        compDiff=100*compVal*valueTemp/aveSize;
        if (m[-1] == 0)
        {
            cc_diff(ptr-3,lTemp,aTemp,bTemp,t);
            ws_push(int(round(t +compDiff)), mofs - 1, iofs - 3, compValStep);
            active_queue = ws_min(active_queue, int(round(t +compDiff ))); // check if queue of this element is prior to the current queue (and should be proceeded in the next iteration)
            m[-1] = IN_QUEUE; // mark in mask as in a queue
        }
        if (m[1] == 0)
        {
            cc_diff(ptr+3,lTemp,aTemp,bTemp,t);
            ws_push(int(round(t+compDiff)), mofs + 1, iofs + 3, compValStep);
            active_queue = ws_min(active_queue, int(round(t +compDiff)));
            m[1] = IN_QUEUE;
        }
        if (m[-mstep] == 0)
        {
            cc_diff(ptr - istep,lTemp,aTemp,bTemp,t);
            ws_push(int(round(t +compDiff)), mofs - mstep, iofs - istep, compValStep);
            active_queue = ws_min(active_queue, int(round(t +compDiff )));
            m[-mstep] = IN_QUEUE;
        }
        if (m[mstep] == 0)
        {
            cc_diff(ptr + istep,lTemp,aTemp,bTemp,t);
            ws_push(int(round(t +compDiff)), mofs + mstep, iofs + istep,  compValStep);
            active_queue = ws_min(active_queue, int(round(t +compDiff )));
            m[mstep] = IN_QUEUE;
        }
    }
    
    
    
    for(i=1;i<imgrow-1;i++)
    {
        for(j=1;j < imgcol-1; j++)
        {
            if(mask[i*imgcol+j]<1)
            {
                int* m = mask+i*imgcol+j;
                uchar* ptr = img + 3*i*imgcol+j * 3;
                int idx=2000, t,tt=0;
                
                if (m[-1] > 0)
                {
                    c_diff(ptr, ptr - 3, t);
                    if (idx>t)
                    {
                        idx=t;
                        tt=m[-1];
                    }
                }
                if (m[1] > 0)
                {
                    c_diff(ptr, ptr + 3, t);
                    if (idx>t)
                    {
                        idx=t;
                        tt=m[1];
                    }
                }
                if (m[-mstep] > 0)
                {
                    c_diff(ptr, ptr - istep, t);
                    if (idx>t)
                    {
                        idx=t;
                        tt=m[-mstep];
                    }
                }
                if (m[mstep] > 0)
                {
                    c_diff(ptr, ptr + istep, t);
                    if (idx>t)
                    {
                        idx=t;
                        tt=m[mstep] ;
                    }
                }
                m[0]=tt;
                labelLab[tt*labstep-3]+=ptr[0];
                labelLab[tt*labstep-2]+=ptr[1];
                labelLab[tt*labstep-1]+=ptr[2];
                numberLab[tt-1]++;
            }
        }
        
    }
    labelnumber[0]=labelId;
    int  label1=labelId;
    
    
    
    if (postProcessing==1)
    {
        int labelpri;
        CvMat *cvGrey = cvCreateMat(imgrow,imgcol,CV_8UC1);
        uchar *ptrGrey;
        Mat imageGray;
        Mat imgColor=imgRGB;
        GaussianBlur( imgRGB, imgRGB, Size(3,3), 0, 0, BORDER_DEFAULT );
        cvtColor(imgRGB,imageGray,CV_BGR2GRAY);
        CvMat ImgGrey = imageGray;
        cvCopy(& ImgGrey, cvGrey);
        ptrGrey = cvGrey->data.ptr;
        
        
        cv::Mat GradientMat=Mat::zeros(imgrow, imgcol, CV_32SC1);
        CvMat *cvGradient = cvCreateMat(imgrow,imgcol,CV_8UC1);
        uchar *ptrGradient;
        Mat grad;
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        
        Sobel( imageGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        Sobel( imageGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        CvMat tempgrad =grad;
        cvCopy(& tempgrad, cvGradient);
        ptrGradient=cvGradient->data.ptr;
        int gg=6;
        
        
        
        for(i=0;i<imgrow;i++)
        {
            for(j=0;j < imgcol ; j++)
            {
                iflabeled[i*imgcol+j]=0;
            }
        }
        
        for (int itr=1; itr<itrSet+1;itr++)
        {
            
            
            int tagL=0-itr;
            for(i=1;i<imgrow-1;i++)
            {
                for(j=1;j < imgcol-1; j++)
                {
                    int maskvalue=mask[i*imgcol+j];
                    
                    if (mask[i*imgcol+j+1]>maskvalue)
                    {
                        active_queue=0;
                        ws_push(active_queue,i*imgcol+j, maskvalue, 0);
                        iflabeled[i*imgcol+j]=tagL;
                    }
                    else if (mask[i*imgcol+j-1]>maskvalue)
                    {
                        active_queue=0;
                        ws_push(active_queue,i*imgcol+j, maskvalue, 0);
                        iflabeled[i*imgcol+j]=tagL;
                    }
                    else if (mask[(i+1)*imgcol+j]>maskvalue)
                    {
                        active_queue=0;
                        ws_push(active_queue,i*imgcol+j, maskvalue, 0);
                        iflabeled[i*imgcol+j]=tagL;
                    }
                    else if (mask[(i-1)*imgcol+j]>maskvalue)
                    {
                        active_queue=0;
                        ws_push(active_queue,i*imgcol+j, maskvalue, 0);
                        iflabeled[i*imgcol+j]=tagL;
                    }
                }
            }
            
            for (;;)
            {
                int masknow;
                int mofs, valuePri;
                int qqq;
                if (q[active_queue].first == 0)
                {
                    for (qqq = active_queue + 1; qqq < NQ+1; qqq++)
                        if (q[qqq].first)
                            break;
                    if (qqq > NQ)
                        break;
                    active_queue = qqq;
                }
                ws_pop(active_queue, mofs, valuePri, masknow);
                int ifColor=1;
                uchar* ptr;
                ptr = img + 3*mofs;
                
                iflabeled[mofs]=itr;
                int valueNei;
                if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep]!=valuePri&&(mask[mofs-mstep]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs-mstep-1]!=valuePri))
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep]!=valuePri&&(mask[mofs+mstep]!=valuePri||mask[mofs+mstep+1]!=valuePri||mask[mofs+mstep-1]!=valuePri))
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs-1]!=valuePri&&(mask[mofs+mstep+1]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs+1]!=valuePri))
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&(mask[mofs+mstep-1]!=valuePri||mask[mofs-mstep-1]!=valuePri||mask[mofs-1]!=valuePri))
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs+mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep-1]!=valuePri)
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs+mstep+1]!=valuePri)
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs-mstep+1]!=valuePri)
                {
                    iflabeled[mofs]=0;
                }
                else if (mask[mofs-mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep-1]!=valuePri)
                {
                    iflabeled[mofs]=0;
                }
                else
                {
                    
                    int labT=valuePri;
                    double minT=10000.0000;
                    int numpri=numberLab[valuePri-1]-1;
                    int flagN=0;
                    if(numpri==0)
                    {
                        numpri=1;
                        flagN=1;
                    }
                    int lTemp;
                    int aTemp;
                    int bTemp;
                    double deltapri;
                    if (flagN==1)
                        deltapri=0;
                    else
                    {
                        
                        aTemp=(labelLab[valuePri*labstep-2]-ptr[1])/numpri;
                        bTemp=(labelLab[valuePri*labstep-1]-ptr[2])/numpri;
                        lTemp=(labelLab[valuePri*labstep-3]-ptr[0])/numpri;
                        
                        cc_diff(ptr,lTemp,aTemp,bTemp,deltapri);
                    }
                    valueNei = mask[mofs-1];
                    
                    if (valueNei> 0&&valueNei!=valuePri)
                    {
                        double deltanei;
                        
                        aTemp=labelLab[valueNei*labstep-2]/numberLab[valueNei-1];
                        bTemp=labelLab[valueNei*labstep-1]/numberLab[valueNei-1];
                        lTemp=labelLab[valueNei*labstep-3]/numberLab[valueNei-1];
                        cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                        
                        if (deltanei<deltapri-deltaC&&deltanei<=minT)
                        {
                            labT=valueNei;
                            minT=deltanei;
                        }
                    }
                    valueNei = mask[mofs+1];
                    if (valueNei> 0&&valueNei!=valuePri)
                    {
                        double deltanei;
                        
                        aTemp=labelLab[valueNei*labstep-2]/numberLab[valueNei-1];
                        bTemp=labelLab[valueNei*labstep-1]/numberLab[valueNei-1];
                        lTemp=labelLab[valueNei*labstep-3]/numberLab[valueNei-1];
                        cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                        if (deltanei<deltapri-deltaC&&deltanei<=minT)
                        {
                            labT=valueNei;
                            minT=deltanei;
                        }
                    }
                    valueNei = mask[mofs+mstep];
                    if (valueNei> 0&&valueNei!=valuePri)
                    {
                        double deltanei;
                        aTemp=labelLab[valueNei*labstep-2]/numberLab[valueNei-1];
                        bTemp=labelLab[valueNei*labstep-1]/numberLab[valueNei-1];
                        lTemp=labelLab[valueNei*labstep-3]/numberLab[valueNei-1];
                        cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                        if (deltanei<deltapri-deltaC&&deltanei<=minT)
                        {
                            labT=valueNei;
                            minT=deltanei;
                        }
                    }
                    valueNei = mask[mofs-mstep];
                    if (valueNei> 0&&valueNei!=valuePri)
                    {
                        double deltanei;
                        
                        aTemp=labelLab[valueNei*labstep-2]/numberLab[valueNei-1];
                        bTemp=labelLab[valueNei*labstep-1]/numberLab[valueNei-1];
                        lTemp=labelLab[valueNei*labstep-3]/numberLab[valueNei-1];
                        cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                        if (deltanei<deltapri-deltaC&&deltanei<=minT)
                        {
                            labT=valueNei;
                            minT=deltanei;
                        }
                    }
                    mask[mofs]=labT;
                    
                }
                if (valuePri!=mask[mofs])
                {
                    int labT=mask[mofs];
                    
                    numberLab[valuePri-1]--;
                    numberLab[labT-1]++;
                    labelLab[labT*labstep-1]+=ptr[2];
                    labelLab[labT*labstep-2]+=ptr[1];
                    labelLab[labT*labstep-3]+=ptr[0];
                    labelLab[valuePri*labstep-1]-=ptr[2];
                    labelLab[valuePri*labstep-2]-=ptr[1];
                    labelLab[valuePri*labstep-3]-=ptr[0];
                    valuePri=labT;
                }
                
                if (active_queue<MQ-1)
                {
                    valueNei = mask[mofs-1];
                    if(valueNei!=valuePri&&iflabeled[mofs-1]!=tagL&&iflabeled[mofs-1]!=itr&&valueNei>0)
                    {
                        ws_push(active_queue+1,mofs-1, valueNei, 0);
                        iflabeled[mofs-1]=tagL;
                    }
                    valueNei = mask[mofs+1];
                    if(valueNei!=valuePri&&iflabeled[mofs+1]!=tagL&&iflabeled[mofs+1]!=itr&&valueNei>0)
                    {
                        ws_push(active_queue+1,mofs+1, valueNei, 0);
                        iflabeled[mofs+1]=tagL;
                    }
                    valueNei = mask[mofs-mstep];
                    if(valueNei!=valuePri&&iflabeled[mofs-mstep]!=tagL&&iflabeled[mofs-mstep]!=itr&&valueNei>0)
                    {
                        ws_push(active_queue+1,mofs-mstep, valueNei, 0);
                        iflabeled[mofs-mstep]=tagL;
                    }
                    valueNei = mask[mofs+mstep];
                    if(valueNei!=valuePri&&iflabeled[mofs+mstep]!=tagL&&iflabeled[mofs+mstep]!=itr&&valueNei>0)
                    {
                        ws_push(active_queue+1,mofs+mstep, valueNei, 0);
                        iflabeled[mofs+mstep]=tagL;
                    }
                }
                
            }
            if (deltaC>0)
            {
                for(i=1;i<imgrow-1;i++)
                {
                    for(j=1;j < imgcol-1; j++)
                    {
                        int mofs=i*imgcol+j;
                        int valuePri=mask[mofs];
                        for(;;)
                        {
                            if (mask[mofs+1]==valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                            {
                                
                                
                                if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs+1-mstep]!=valuePri&&(abs(ptrGrey[mofs+1]-ptrGrey[mofs+1+mstep])<gg||abs(ptrGrey[mofs+1]-ptrGrey[mofs+1-mstep])<gg)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                    if (mask[mofs-1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs-1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                    if (del0<del&&mask[mofs-mstep]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs-mstep];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                    if (del0<del&&mask[mofs+mstep]>0)
                                    {
                                        labT=mask[mofs+mstep];
                                    }
                                    mask[mofs]=labT;
                                    
                                    numberLab[valuePri-1]--;
                                    numberLab[labT-1]++;
                                    uchar* ptr;
                                    ptr= img + 3*mofs;
                                    labelLab[labT*labstep-3]+=ptr[0];
                                    labelLab[labT*labstep-2]+=ptr[1];
                                    labelLab[labT*labstep-1]+=ptr[2];
                                    labelLab[valuePri*labstep-3]-=ptr[0];
                                    labelLab[valuePri*labstep-2]-=ptr[1];
                                    labelLab[valuePri*labstep-1]-=ptr[2];
                                    mofs=mofs+1;
                                }
                                else
                                    break;
                                
                                
                            }
                            else if (mask[mofs-1]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                            {
                                if (ptrGradient[mofs]<gg||(mask[mofs-1+mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-1]-ptrGrey[mofs-1+mstep])<gg||abs(ptrGrey[mofs-1]-ptrGrey[mofs-1-mstep])<gg)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                    if (mask[mofs+1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs+1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                    if (del0<del&&mask[mofs-mstep]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs-mstep];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                    if (del0<del&&mask[mofs+mstep]>0)
                                    {
                                        labT=mask[mofs+mstep];
                                    }
                                    mask[mofs]=labT;
                                    
                                    numberLab[valuePri-1]--;
                                    numberLab[labT-1]++;
                                    uchar* ptr;
                                    ptr= img + 3*mofs;
                                    labelLab[labT*labstep-3]+=ptr[0];
                                    labelLab[labT*labstep-2]+=ptr[1];
                                    labelLab[labT*labstep-1]+=ptr[2];
                                    labelLab[valuePri*labstep-3]-=ptr[0];
                                    labelLab[valuePri*labstep-2]-=ptr[1];
                                    labelLab[valuePri*labstep-1]-=ptr[2];
                                    
                                    mofs=mofs-1;
                                }
                                else
                                    break;
                            }
                            else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs+mstep]!=valuePri)
                            {
                                if (ptrGradient[mofs]<gg||(mask[mofs+1-mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-mstep]-ptrGrey[mofs+1-mstep])<gg||abs(ptrGrey[mofs-mstep]-ptrGrey[mofs-1-mstep])<gg)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                    if (mask[mofs-1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs-1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                    if (del0<del&&mask[mofs+1]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs+1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                    if (del0<del&&mask[mofs+mstep]>0)
                                    {
                                        labT=mask[mofs+mstep];
                                    }
                                    mask[mofs]=labT;
                                    
                                    numberLab[valuePri-1]--;
                                    numberLab[labT-1]++;
                                    uchar* ptr;
                                    ptr= img + 3*mofs;
                                    labelLab[labT*labstep-3]+=ptr[0];
                                    labelLab[labT*labstep-2]+=ptr[1];
                                    labelLab[labT*labstep-1]+=ptr[2];
                                    labelLab[valuePri*labstep-3]-=ptr[0];
                                    labelLab[valuePri*labstep-2]-=ptr[1];
                                    labelLab[valuePri*labstep-1]-=ptr[2];
                                    mofs=mofs-mstep;
                                }
                                else
                                    break;
                            }
                            else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri)
                            {
                                if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs-1+mstep]!=valuePri&&(abs(ptrGrey[mofs+mstep]-ptrGrey[mofs+1+mstep])<gg||abs(ptrGrey[mofs+mstep]-ptrGrey[mofs-1+mstep])<gg)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                    if (mask[mofs-1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs-1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                    if (del0<del&&mask[mofs-mstep]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs-mstep];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                    if (del0<del&&mask[mofs+1]>0)
                                    {
                                        labT=mask[mofs+1];
                                    }
                                    mask[mofs]=labT;
                                    
                                    numberLab[valuePri-1]--;
                                    numberLab[labT-1]++;
                                    uchar* ptr;
                                    ptr= img + 3*mofs;
                                    labelLab[labT*labstep-3]+=ptr[0];
                                    labelLab[labT*labstep-2]+=ptr[1];
                                    labelLab[labT*labstep-1]+=ptr[2];
                                    labelLab[valuePri*labstep-3]-=ptr[0];
                                    labelLab[valuePri*labstep-2]-=ptr[1];
                                    labelLab[valuePri*labstep-1]-=ptr[2];
                                    mofs=mofs+mstep;
                                }
                                else
                                    break;
                            }
                            else
                                break;
                        }
                    }
                }
            }
            
            
            
        }
        
        
        
        if (deltaC>0)
        {
            for(i=1;i<imgrow-1;i++)
            {
                for(j=1;j < imgcol-1; j++)
                {
                    int mofs=i*imgcol+j;
                    int valuePri=mask[mofs];
                    for(;;)
                    {
                        if (mask[mofs+1]==valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                        {
                            
                            
                            if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs+1-mstep]!=valuePri&&(abs(ptrGrey[mofs+1]-ptrGrey[mofs+1+mstep])<gg||abs(ptrGrey[mofs+1]-ptrGrey[mofs+1-mstep])<gg)))
                            {
                                int del=10000;
                                int labT;
                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                if (mask[mofs-1]>0&&del0<del)
                                {
                                    del=del0;
                                    labT=mask[mofs-1];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                if (del0<del&&mask[mofs-mstep]>0)
                                {
                                    del=del0;
                                    labT=mask[mofs-mstep];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                if (del0<del&&mask[mofs+mstep]>0)
                                {
                                    labT=mask[mofs+mstep];
                                }
                                mask[mofs]=labT;
                                
                                numberLab[valuePri-1]--;
                                numberLab[labT-1]++;
                                uchar* ptr;
                                ptr= img + 3*mofs;
                                labelLab[labT*labstep-3]+=ptr[0];
                                labelLab[labT*labstep-2]+=ptr[1];
                                labelLab[labT*labstep-1]+=ptr[2];
                                labelLab[valuePri*labstep-3]-=ptr[0];
                                labelLab[valuePri*labstep-2]-=ptr[1];
                                labelLab[valuePri*labstep-1]-=ptr[2];
                                mofs=mofs+1;
                            }
                            else
                                break;
                            
                            
                        }
                        else if (mask[mofs-1]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                        {
                            if (ptrGradient[mofs]<gg||(mask[mofs-1+mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-1]-ptrGrey[mofs-1+mstep])<gg||abs(ptrGrey[mofs-1]-ptrGrey[mofs-1-mstep])<gg)))
                            {
                                int del=10000;
                                int labT;
                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                if (mask[mofs+1]>0&&del0<del)
                                {
                                    del=del0;
                                    labT=mask[mofs+1];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                if (del0<del&&mask[mofs-mstep]>0)
                                {
                                    del=del0;
                                    labT=mask[mofs-mstep];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                if (del0<del&&mask[mofs+mstep]>0)
                                {
                                    labT=mask[mofs+mstep];
                                }
                                mask[mofs]=labT;
                                
                                numberLab[valuePri-1]--;
                                numberLab[labT-1]++;
                                uchar* ptr;
                                ptr= img + 3*mofs;
                                labelLab[labT*labstep-3]+=ptr[0];
                                labelLab[labT*labstep-2]+=ptr[1];
                                labelLab[labT*labstep-1]+=ptr[2];
                                labelLab[valuePri*labstep-3]-=ptr[0];
                                labelLab[valuePri*labstep-2]-=ptr[1];
                                labelLab[valuePri*labstep-1]-=ptr[2];
                                mofs=mofs-1;
                            }
                            else
                                break;
                        }
                        else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs+mstep]!=valuePri)
                        {
                            if (ptrGradient[mofs]<gg||(mask[mofs+1-mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-mstep]-ptrGrey[mofs+1-mstep])<gg||abs(ptrGrey[mofs-mstep]-ptrGrey[mofs-1-mstep])<gg)))
                            {
                                int del=10000;
                                int labT;
                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                if (mask[mofs-1]>0&&del0<del)
                                {
                                    del=del0;
                                    labT=mask[mofs-1];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                if (del0<del&&mask[mofs+1]>0)
                                {
                                    del=del0;
                                    labT=mask[mofs+1];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                if (del0<del&&mask[mofs+mstep]>0)
                                {
                                    labT=mask[mofs+mstep];
                                }
                                mask[mofs]=labT;
                                
                                numberLab[valuePri-1]--;
                                numberLab[labT-1]++;
                                uchar* ptr;
                                ptr= img + 3*mofs;
                                labelLab[labT*labstep-3]+=ptr[0];
                                labelLab[labT*labstep-2]+=ptr[1];
                                labelLab[labT*labstep-1]+=ptr[2];
                                labelLab[valuePri*labstep-3]-=ptr[0];
                                labelLab[valuePri*labstep-2]-=ptr[1];
                                labelLab[valuePri*labstep-1]-=ptr[2];
                                mofs=mofs-mstep;
                            }
                            else
                                break;
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri)
                        {
                            if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs-1+mstep]!=valuePri&&(abs(ptrGrey[mofs+mstep]-ptrGrey[mofs+1+mstep])<gg||abs(ptrGrey[mofs+mstep]-ptrGrey[mofs-1+mstep])<gg)))
                            {
                                int del=10000;
                                int labT;
                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                if (mask[mofs-1]>0&&del0<del)
                                {
                                    del=del0;
                                    labT=mask[mofs-1];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                if (del0<del&&mask[mofs-mstep]>0)
                                {
                                    del=del0;
                                    labT=mask[mofs-mstep];
                                }
                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                if (del0<del&&mask[mofs+1]>0)
                                {
                                    labT=mask[mofs+1];
                                }
                                mask[mofs]=labT;
                                
                                numberLab[valuePri-1]--;
                                numberLab[labT-1]++;
                                uchar* ptr;
                                ptr= img + 3*mofs;
                                labelLab[labT*labstep-3]+=ptr[0];
                                labelLab[labT*labstep-2]+=ptr[1];
                                labelLab[labT*labstep-1]+=ptr[2];
                                labelLab[valuePri*labstep-3]-=ptr[0];
                                labelLab[valuePri*labstep-2]-=ptr[1];
                                labelLab[valuePri*labstep-1]-=ptr[2];
                                mofs=mofs+mstep;
                            }
                            else
                                break;
                        }
                        else
                            break;
                    }
                }
            }
        }
        
        
        
        
        int LBM=1;
        if (deltaC==0)
            LBM=0;
        if (LBM==1)
            
        {
            
            cv::Mat PositionMat0=Mat::zeros(labelId, 2, CV_32SC1);
            CvMat *cvPosition0 = cvCreateMat(labelId,2,CV_32SC1);
            int *ptrPosition0;
            CvMat Position0=PositionMat0;
            cvCopy(& Position0, cvPosition0);
            ptrPosition0=cvPosition0->data.i;
            
            for(i=1;i<imgrow-1;i++)
            {
                for(j=1;j < imgcol-1; j++)
                {
                    int* m;
                    m=mask+i*imgcol+j;
                    int labelpri=m[0];
                    ptrPosition0[2*labelpri-2]+=i;
                    ptrPosition0[2*labelpri-1]+=j;
                    
                }
            }
            
            
            int gth=6;
            gg=6;
            
            int SQ=disAver;
            int itrr=2;
            for(int rr=0;rr<itrr;rr++)
            {
                cv::Mat SNeighborMat0=Mat::zeros(labelId, labelId, CV_32SC1);
                CvMat *cvSNeighbor0 = cvCreateMat(labelId,labelId,CV_32SC1);
                int *ptrSNeighbor0;
                CvMat SNeighbor0=SNeighborMat0;
                cvCopy(& SNeighbor0, cvSNeighbor0);
                ptrSNeighbor0=cvSNeighbor0->data.i;
                
                cv::Mat SNumberMat0=Mat::zeros(labelId, labelId, CV_32SC1);
                CvMat *cvSNumber0 = cvCreateMat(labelId,labelId,CV_32SC1);
                int *ptrSNumber0;
                CvMat SNumber0=SNumberMat0;
                cvCopy(& SNumber0, cvSNumber0);
                ptrSNumber0=cvSNumber0->data.i;
                for(i=1;i<imgrow-1;i++)
                {
                    for(j=1;j < imgcol-1; j++)
                    {
                        int* m;
                        m=mask+i*imgcol+j;
                        int labelpri=m[0];
                        if (m[1]!=labelpri&&m[1]>0)
                        {
                            int labelnew=m[1];
                            ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                            ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                            
                        }
                        else if (m[-1]!=labelpri&&m[-1]>0)
                        {
                            int labelnew=m[-1];
                            ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                            ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                            
                        }
                        else if (m[mstep]!=labelpri&&m[mstep]>0)
                        {
                            int labelnew=m[mstep];
                            ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                            ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                            
                        }
                        else if (m[-mstep]!=labelpri&&m[-mstep]>0)
                        {
                            int labelnew=m[-mstep];
                            ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                            ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                            
                        }
                        
                    }
                }
                
                
                
                for(i=0;i<labelId;i++)
                {
                    for(j=0;j < labelId; j++)
                    {
                        if (ptrSNeighbor0[i*labelId+j]>0)
                        {
                            int mofs=i*labelId+j;
                            ptrSNeighbor0[mofs]=ptrSNeighbor0[mofs]/ptrSNumber0[mofs];
                            if (ptrSNeighbor0[mofs]>gth)
                                ptrSNeighbor0[mofs]=1;
                            else
                                ptrSNeighbor0[mofs]=0;
                        }
                        
                    }
                }
                
                
                for(i=0;i<imgrow;i++)
                {
                    for(j=0;j < imgcol ; j++)
                    {
                        iflabeled[i*imgcol+j]=0;
                    }
                }
                for(i=0;i<imgrow;i++)
                {
                    iflabeled[i*imgcol]=-2;
                    iflabeled[i*imgcol+imgcol-1]=-2;
                }
                for(j=0;j < imgcol; j++)
                {
                    iflabeled[j]=-2;
                    iflabeled[imgrow*imgcol-imgcol+j]=-2;
                }
                
                
                int mofs;
                int valuePri;
                int labelpri;
                int tagL=rr+1;
                active_queue=0;
                for(i=1;i<imgrow-1;i++)
                {
                    for(j=1;j < imgcol-1; j++)
                    {
                        int* m;
                        m=mask+i*imgcol+j;
                        labelpri=m[0];
                        int mp=i*imgcol+j;
                        mofs=mp;
                        valuePri=labelpri;
                        
                        
                        if (m[1]>labelpri&&m[1]>0&&iflabeled[mp]!=tagL)
                        {
                            int labelnew=m[1];
                            if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=-1;
                            }
                        }
                        else if (m[-1]>labelpri&&m[-1]>0&&iflabeled[mp]!=tagL)
                        {
                            int labelnew=m[-1];
                            if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=-1;
                            }
                        }
                        else if (m[mstep]>labelpri&&m[mstep]>0&&iflabeled[mp]!=tagL)
                        {
                            int labelnew=m[mstep];
                            if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=-1;
                            }
                        }
                        else if (m[-mstep]>labelpri&&m[-mstep]>0&&iflabeled[mp]!=tagL)
                        {
                            int labelnew=m[-mstep];
                            if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=-1;
                            }
                        }
                    }
                }
                
                
                for (;;)
                {
                    if (q[active_queue].first == 0)
                    {
                        int qqq;
                        for (qqq = active_queue+1; qqq < SQ+1; qqq++)
                            if (q[qqq].first)
                                break;
                        if (qqq > SQ)
                            break;
                        active_queue = qqq;
                    }
                    int i_ori,j_ori;
                    ws_pop(active_queue, mofs, i_ori, j_ori);
                    int valuePri=mask[mofs];
                    int labelpri=valuePri;
                    
                    
                    if(0<1)
                    {
                        if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep]!=valuePri&&(mask[mofs-mstep]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs-mstep-1]!=valuePri))
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep]!=valuePri&&(mask[mofs+mstep]!=valuePri||mask[mofs+mstep+1]!=valuePri||mask[mofs+mstep-1]!=valuePri))
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs-1]!=valuePri&&(mask[mofs+mstep+1]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs+1]!=valuePri))
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&(mask[mofs+mstep-1]!=valuePri||mask[mofs-mstep-1]!=valuePri||mask[mofs-1]!=valuePri))
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep-1]!=valuePri)
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs+mstep+1]!=valuePri)
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs-mstep+1]!=valuePri)
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (mask[mofs-mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep-1]!=valuePri)
                        {
                            if(active_queue<SQ-1)
                            {
                                ws_push(active_queue+1,mofs, i_ori, j_ori);
                                iflabeled[mofs]=-1;
                            }
                        }
                        else if (labelpri>0)
                        {
                            double minT=100000000000000000;
                            int labT=labelpri;
                            int flag=0;
                            if (mask[mofs+mstep]!=labelpri&&mask[mofs+mstep]>0)
                            {
                                int labelnew=mask[mofs+mstep];
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                                {
                                    flag=1;
                                    
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(numberLab[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(numberLab[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(numberLab[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(numberLab[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                                
                            }
                            if (mask[mofs-mstep]!=labelpri&&mask[mofs-mstep]>0)
                            {
                                int labelnew=mask[mofs-mstep];
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                                {
                                    flag=1;
                                    
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(numberLab[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(numberLab[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(numberLab[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(numberLab[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                            }
                            if (mask[mofs+1]!=labelpri&&mask[mofs+1]>0)
                            {
                                int labelnew=mask[mofs+1];
                                
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                                {
                                    flag=1;
                                    
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(numberLab[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(numberLab[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(numberLab[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(numberLab[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                            }
                            if (mask[mofs-1]!=labelpri&&mask[mofs-1]>0)
                            {
                                int labelnew=mask[mofs-1];
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==0&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==0)
                                {
                                    flag=1;
                                    
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(numberLab[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(numberLab[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(numberLab[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(numberLab[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                            }
                            
                            if (labT!=labelpri)
                            {
                                mask[mofs]=labT;
                                numberLab[labelpri-1]--;
                                numberLab[labT-1]++;
                                ptrPosition0[2*labelpri-2]-=i_ori;
                                ptrPosition0[2*labelpri-1]-=j_ori;
                                ptrPosition0[2*labT-2]+=i_ori;
                                ptrPosition0[2*labT-1]+=j_ori;
                                iflabeled[mofs]=0;
                                labelpri=labT;
                            }
                            else
                            {
                                iflabeled[mofs]+=2;
                            }
                            if(active_queue<SQ-1)
                            {
                                if (mask[mofs+1]!=labelpri&&mask[mofs+1]>0&&iflabeled[mofs+1]>-1&&iflabeled[mofs+1]<100)
                                {
                                    ws_push(active_queue+1,mofs+1, i_ori, j_ori+1);
                                    if (iflabeled[mofs+1]==0)
                                        iflabeled[mofs+1]=-1;
                                }
                                if (mask[mofs-1]!=labelpri&&mask[mofs-1]>0&&iflabeled[mofs-1]>-1&&iflabeled[mofs-1]<100)
                                {
                                    ws_push(active_queue+1,mofs-1, i_ori, j_ori-1);
                                    if (iflabeled[mofs-1]==0)
                                        iflabeled[mofs-1]=-1;
                                }
                                if (mask[mofs+mstep]!=labelpri&&mask[mofs+mstep]>0&&iflabeled[mofs+mstep]>-1&&iflabeled[mofs+mstep]<100)
                                {
                                    ws_push(active_queue+1,mofs+mstep, i_ori+1, j_ori);
                                    if (iflabeled[mofs+mstep]==0)
                                        iflabeled[mofs+mstep]=-1;
                                }
                                if (mask[mofs-mstep]!=labelpri&&mask[mofs-mstep]>0&&iflabeled[mofs-mstep]>-1&&iflabeled[mofs-mstep]<100)
                                {
                                    ws_push(active_queue+1,mofs-mstep, i_ori-1, j_ori);
                                    if (iflabeled[mofs-mstep]==0)
                                        iflabeled[mofs-mstep]=-1;
                                }
                            }
                        }
                    }
                }
                
                
                if (deltaC>0)
                {
                    for(i=1;i<imgrow-1;i++)
                    {
                        for(j=1;j < imgcol-1; j++)
                        {
                            int mofs=i*imgcol+j;
                            int i_ori=i;
                            int j_ori=j;
                            
                            for(;;)
                            {
                                int ggg=0;
                                
                                int valuePri=mask[mofs];
                                
                                int ifChanged=0;
                                if (mask[mofs+1]==valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-1]>0&&mask[mofs-mstep]!=valuePri&&mask[mofs-mstep]>0&&mask[mofs+mstep]!=valuePri&&mask[mofs+mstep]>0)
                                {
                                    
                                    int del=10000000000;
                                    int labT=valuePri;
                                    int valueNew=mask[mofs-1];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        labT=valueNew;
                                        del=disT;
                                    }
                                    valueNew=mask[mofs-mstep];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        if (disT<del)
                                        {
                                            labT=valueNew;
                                            del=disT;
                                        }
                                    }
                                    valueNew=mask[mofs+mstep];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        if (disT<del)
                                        {
                                            labT=valueNew;
                                        }
                                    }
                                    if (labT!=valuePri)
                                    {
                                        mask[mofs]=labT;
                                        numberLab[valuePri-1]--;
                                        numberLab[labT-1]++;
                                        ptrPosition0[2*valuePri-2]=ptrPosition0[2*valuePri-2]-i_ori;
                                        ptrPosition0[2*valuePri-1]=ptrPosition0[2*valuePri-1]-j_ori;
                                        ptrPosition0[2*labT-2]=ptrPosition0[2*labT-2]+i_ori;
                                        ptrPosition0[2*labT-1]=ptrPosition0[2*labT-1]+j_ori;
                                        j_ori+=1;
                                        mofs=mofs+1;
                                        ifChanged=1;
                                    }
                                    
                                }
                                
                                else if (mask[mofs-1]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs+1]>0&&mask[mofs-mstep]!=valuePri&&mask[mofs-mstep]>0&&mask[mofs+mstep]!=valuePri&&mask[mofs+mstep]>0)
                                    
                                {int del=10000000000;
                                 int labT=valuePri;
                                 int valueNew=mask[mofs+1];
                                 if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                 {
                                     double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                     double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                     double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                     labT=valueNew;
                                     del=disT;
                                 }
                                 valueNew=mask[mofs-mstep];
                                 if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                 {
                                     double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                     double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                     double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                     if (disT<del)
                                     {
                                         labT=valueNew;
                                         del=disT;
                                     }
                                 }
                                 valueNew=mask[mofs+mstep];
                                 if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                 {
                                     double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                     double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                     double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                     if (disT<del)
                                     {
                                         labT=valueNew;
                                     }
                                 }
                                 if (labT!=valuePri)
                                 {
                                     mask[mofs]=labT;
                                     numberLab[valuePri-1]--;
                                     numberLab[labT-1]++;
                                     ptrPosition0[2*valuePri-2]=ptrPosition0[2*valuePri-2]-i_ori;
                                     ptrPosition0[2*valuePri-1]=ptrPosition0[2*valuePri-1]-j_ori;
                                     ptrPosition0[2*labT-2]=ptrPosition0[2*labT-2]+i_ori;
                                     ptrPosition0[2*labT-1]=ptrPosition0[2*labT-1]+j_ori;
                                     j_ori-=1;
                                     mofs=mofs-1;
                                     ifChanged=1;
                                 }
                                }
                                
                                
                                else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs+1]>0&&mask[mofs-1]!=valuePri&&mask[mofs-1]>0&&mask[mofs+mstep]!=valuePri&&mask[mofs+mstep]>0)
                                {
                                    int del=10000000000;
                                    int labT=valuePri;
                                    int valueNew=mask[mofs-1];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        labT=valueNew;
                                        del=disT;
                                    }
                                    valueNew=mask[mofs+1];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        if (disT<del)
                                        {
                                            labT=valueNew;
                                            del=disT;
                                        }
                                    }
                                    valueNew=mask[mofs+mstep];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        if (disT<del)
                                        {
                                            labT=valueNew;
                                        }
                                    }
                                    if (labT!=valuePri)
                                    {
                                        mask[mofs]=labT;
                                        numberLab[valuePri-1]--;
                                        numberLab[labT-1]++;
                                        ptrPosition0[2*valuePri-2]=ptrPosition0[2*valuePri-2]-i_ori;
                                        ptrPosition0[2*valuePri-1]=ptrPosition0[2*valuePri-1]-j_ori;
                                        ptrPosition0[2*labT-2]=ptrPosition0[2*labT-2]+i_ori;
                                        ptrPosition0[2*labT-1]=ptrPosition0[2*labT-1]+j_ori;
                                        i_ori-=1;
                                        mofs=mofs-mstep;
                                        ifChanged=1;
                                    }
                                    
                                }
                                else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs+1]>0&&mask[mofs-1]!=valuePri&&mask[mofs-1]>0&&mask[mofs-mstep]!=valuePri&&mask[mofs-mstep]>0)
                                {
                                    int del=10000000000;
                                    int labT=valuePri;
                                    int valueNew=mask[mofs-1];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        labT=valueNew;
                                        del=disT;
                                    }
                                    valueNew=mask[mofs-mstep];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        if (disT<del)
                                        {
                                            labT=valueNew;
                                            del=disT;
                                        }
                                    }
                                    valueNew=mask[mofs+1];
                                    if (valueNew>0&&(ptrGradient[mofs]<gg||(ptrSNeighbor0[(valuePri-1)*labelId+valueNew-1]==0&&ptrSNeighbor0[(valueNew-1)*labelId+valuePri-1]==0)||abs(ptrGrey[mofs-1]-ptrGrey[mofs])<ggg))
                                    {
                                        double i_new=double(ptrPosition0[2*valueNew-2])/double(numberLab[valueNew-1]);
                                        double j_new=double(ptrPosition0[2*valueNew-1])/double(numberLab[valueNew-1]);
                                        double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                        if (disT<del)
                                        {
                                            labT=valueNew;
                                        }
                                    }
                                    if (labT!=valuePri)
                                    {
                                        mask[mofs]=labT;
                                        numberLab[valuePri-1]--;
                                        numberLab[labT-1]++;
                                        ptrPosition0[2*valuePri-2]=ptrPosition0[2*valuePri-2]-i_ori;
                                        ptrPosition0[2*valuePri-1]=ptrPosition0[2*valuePri-1]-j_ori;
                                        ptrPosition0[2*labT-2]=ptrPosition0[2*labT-2]+i_ori;
                                        ptrPosition0[2*labT-1]=ptrPosition0[2*labT-1]+j_ori;
                                        i_ori+=1;
                                        mofs=mofs+mstep;
                                        ifChanged=1;
                                    }
                                    
                                }
                                if (ifChanged==0)
                                    break;
                            }
                        }
                    }
                }
                
                cvReleaseMat(&cvSNeighbor0);
                cvReleaseMat(&cvSNumber0);
            }
            cvReleaseMat(&cvPosition0);
            
            
        }
        cvReleaseMat(&cvGrey);
    }
    
    
    for(i=0;i<imgrow;i++)
    {
        for(j=0;j < imgcol ; j++)
        {
            iflabeled[i*imgcol+j]=0;
        }
    }
    for(i=0;i<imgrow;i++)
    {
        iflabeled[i*imgcol]=-3;
        iflabeled[i*imgcol+imgcol-1]=-3;
    }
    for(j=0;j < imgcol; j++)
    {
        iflabeled[j]=-3;
        iflabeled[imgrow*imgcol-imgcol+j]=-3;
    }
    int labelnewID=0;
    for(i=1;i<imgrow-1;i++)
    {
        for(j=1;j < imgcol-1; j++)
        {
            if (iflabeled[i*imgcol+j]==0&&mask[i*imgcol+j]>0)
            {
                int maskvalue=mask[i*imgcol+j];
                labelnewID++;
                active_queue=0;
                ws_push(active_queue,i*imgcol+j, labelnewID, maskvalue);
                iflabeled[i*imgcol+j]=-4;
                for (;;)
                {
                    int masknow;
                    int mofs, value;
                    if (q[active_queue].first == 0)
                        break;
                    
                    ws_pop(active_queue, mofs, value, masknow);
                    mask[mofs]=value;
                    iflabeled[mofs]=1;
                    if (iflabeled[mofs+1]==0&&mask[mofs+1]==masknow)
                    {
                        ws_push(active_queue,mofs+1, value, masknow);
                        iflabeled[mofs+1]=-4;
                    }
                    if (iflabeled[mofs-1]==0&&mask[mofs-1]==masknow)
                    {
                        ws_push(active_queue,mofs-1, value, masknow);
                        iflabeled[mofs-1]=-4;
                    }
                    if (iflabeled[mofs+mstep]==0&&mask[mofs+mstep]==masknow)
                    {
                        ws_push(active_queue,mofs+mstep, value, masknow);
                        iflabeled[mofs+mstep]=-4;
                    }
                    if (iflabeled[mofs-mstep]==0&&mask[mofs-mstep]==masknow)
                    {
                        ws_push(active_queue,mofs-mstep, value, masknow);
                        iflabeled[mofs-mstep]=-4;
                    }
                }
            }
        }
    }
    
    
    labelId=labelnewID;
    labelnumber[0]=labelId;
    
    for(i=0;i<imgrow;i++)
    {
        mask[i*imgcol]=mask[i*imgcol+1];
        mask[i*imgcol+imgcol-1]=mask[i*imgcol+imgcol-2];
    }
    for(j=0;j < imgcol; j++)
    {
        mask[j]=mask[j+mstep];
        mask[imgrow*imgcol-imgcol+j]=mask[imgrow*imgcol-imgcol+j-mstep];
    }
    mask[0]=mask[mstep+1];
    mask[imgrow*imgcol-1]=mask[imgrow*imgcol-2-mstep];
    mask[imgcol-1]=mask[imgcol+mstep-2];
    mask[imgrow*imgcol-imgcol]=mask[imgrow*imgcol-imgcol+1-mstep];
    BMat = Mat(dst, true);
    
    
    cvReleaseMat(&src);
    cvReleaseMat(&dst);
    cvReleaseMat(&labellabptr);
    cvReleaseMat(&numberlabptr);
    cvReleaseMat(&iflabel);
}




void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // ============ parse input ==============
    // create opencv Mat from argument
    
    
    cv::Mat I;
    convertMx2Mat(prhs[0], I);
    int spn = (int)(mxGetScalar(prhs[1])); // number of segments
    double compVal = (double)(mxGetScalar(prhs[2])); // compactness parameter
    //int n = (int)(mxGetScalar(prhs[3]));
    int postprocessing = (int)(mxGetScalar(prhs[3]));
    //int SplitMerge = (int)(mxGetScalar(prhs[5]));
    int ItrSet = (int)(mxGetScalar(prhs[4]));
    //int MQ = (int)(mxGetScalar(prhs[5]));
    int DeltaC = (int)(mxGetScalar(prhs[5]));
    int seedsType = (int)(mxGetScalar(prhs[6]));
    
    
    Mat seeds;
    
    
    
    
    int * labelnumber;
    labelnumber=new int[1];
    // ================== process ================
    cv::Mat B;
    compact_watershed(I, B, compVal, seeds, spn, postprocessing,ItrSet, DeltaC,seedsType,labelnumber);
    
    
    
    // ================ create output ================
    
    
    
    
    
    
    if( nlhs>0)
    {
        convertMat2Mx(B, plhs[0]);
    }
    
    int* spnumber;
    plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    
    
    
    spnumber = (int*)mxGetData(plhs[1]);//gives a void*, cast it to int*
    *spnumber=labelnumber[0];
    
    
    
    
}

//// mexOpenCV mex_WSGL.cpp mex_helper_WSGL.cpp
