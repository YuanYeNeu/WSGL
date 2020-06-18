% This version is modified to enable the Phase 1 and Phase 2 separately,
% when BM is set as 1, the results are of the whole procedure. otherwise
% the results are of phase one only.
% Also this version is modified to enable the global boundary marching and
% local boundary marching separately. If the parameter of LBM in
% mex_WSGL.cpp is set as 1, the local boundary marching is enabled.





clear all;
close all;

currentFolder = pwd;
addpath(genpath(currentFolder));
scalespn=0.01;
Tspn=0;
if Tspn==0
    spn=600;
end


CompactnessCW=0.5; %lambda
BM=1;%0=no;1=yes
ItrSet=2; %Itr
DeltaC=2; %delta
seedsType=1;%0=Square;1=Hexagonal
Files = dir(strcat('img\\','*.jpg'));

time=0.00000;
LengthFiles = length(Files);
for i = 1:LengthFiles;
    img = imread(strcat('img\',Files(i).name));
    str=['img\',Files(i).name];
    
    Row = size(img, 1); Column = size(img, 2);
    
    
    img=uint8(img);
    
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    R=uint8(R);
    G=uint8(G);
    B=uint8(B);
    
    if Tspn==1
        spn=Row*Column*scalespn;
    end
    st = clock;
    [label, spnumber] = mex_WSGL(img, spn, CompactnessCW,BM,ItrSet,DeltaC,seedsType);
    
    deltatime=etime(clock,st);
    time=time+deltatime;
    fprintf(' spnumber=%d\n',spnumber);
   
    labelFinal=label;
   
    
    resultfile= strrep(str,'jpg', 'csv');
    csvwrite(resultfile,labelFinal);
    
    
    seeds=labelFinal;
    
    seedstemp=zeros(Row, Column);
    seedsflag=zeros(Row, Column);
    for j=2:Row-1
        for k=2:Column-1
            if seeds(j,k)~=seeds(j-1,k)
                if seedsflag(j-1,k)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
            if seeds(j,k)~=seeds(j+1,k)
                if seedsflag(j+1,k)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
            if seeds(j,k)~=seeds(j,k-1)
                if seedsflag(j,k-1)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
            if seeds(j,k)~=seeds(j,k+1)
                if seedsflag(j,k+1)==0
                    seedstemp(j,k)=1;
                    seedsflag(j,k)=1;
                end
            end
        end
    end
    seeds=seedstemp;
    img = im2double(img);
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    R(seeds>0)=255;
    G(seeds>0)=0;
    B(seeds>0)=0;
    img(:,:,1) = R;
    img(:,:,2) = G;
    img(:,:,3) = B;
    resultImage = [str '_WSGL.png'];
    imwrite(img,resultImage);
    
    
    
    
    
    
end
avertime=time/LengthFiles;
fprintf(' Average took %.5f second\n',avertime);

