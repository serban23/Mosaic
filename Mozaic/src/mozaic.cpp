#include <iostream>
#include <opencv2/opencv.hpp>
#include "mozaic.h"
using namespace std;
using namespace cv;

#include <filesystem>
using namespace std::filesystem;

///GENEARE
void generateTiles()
{
    vector<int> steps = {0, 32, 64, 96, 128, 160, 192, 224, 255};
    for (int r : steps)
        for (int g : steps)
            for (int b : steps)
            {
                Mat tile(50,50,CV_8UC3, Scalar(b,g,r));
                string name="color_"+to_string(r)+"_"+to_string(g)+"_"+to_string(b)+".bmp";
                imwrite("C:/FACULTATE/3.2/PI/PROIECT/Mozaic/images/tiles/"+name,tile);
            }
}

///LOAD
images loadImages(const string& originalName)
{
    string path = "C:\\FACULTATE\\3.2\\PI\\PROIECT\\Mozaic\\images\\" + originalName + ".bmp";
    Mat source = imread(path, IMREAD_COLOR);

    if (source.empty())
        throw runtime_error("Eroare: nu am putut încărca imaginea originală.");

    resize(source, source, Size(800, 800));
    imshow ("Original", source);

    vector<Mat> tiles;

    for (const auto& entry: directory_iterator("C:\\FACULTATE\\3.2\\PI\\PROIECT\\Mozaic\\images\\tiles")){
        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        if (!img.empty()) {
            //redimensionarea imaginilor mici (nu e nevoie momentan)
            //resize(img, img, Size(50, 50));
            tiles.push_back(img);
        }
    }

    return {source,tiles};
}

///SEGMENTARE
segments imageSegmentation(const Mat& source, int noSegments)
{
    int rows = source.rows;
    int cols = source.cols;

    int grid = sqrt(noSegments);

    if(grid * 50 != rows || grid*50!=cols)
        throw runtime_error("Eroare: dimensiunile imaginii nu sunt compatibile cu numarul de segmente.");

    segments seg;

    for(int i = 0; i<grid; i++)
        for(int j=0;j<grid;j++)
        {
            int x = j*50;
            int y = i*50;

            seg.xs.push_back(x);
            seg.ys.push_back(y);

            Rect r(x,y,50,50);
            seg.s.push_back(source(r).clone());
        }

    return seg;
}

///FUNCTII PENTRU MEDIA DE CULORI
Scalar computeMeans(Mat img)
{
    float meanB=0, meanG=0, meanR=0;
    int totalPixels = img.rows*img.cols;

    for(int i=0;i<img.rows; i++)
        for(int j=0;j<img.cols;j++)
        {
            meanB += img.at<Vec3b>(i,j)[0];
            meanG += img.at<Vec3b>(i,j)[1];
            meanR += img.at<Vec3b>(i,j)[2];
        }
    meanB=meanB/totalPixels;
    meanG=meanG/totalPixels;
    meanR=meanR/totalPixels;
    return{meanR,meanG,meanB};
}

float compareMeans(Scalar segMean, Scalar tileMean)
{
    //distanta euclidiana
    float distB = segMean[0]-tileMean[0];
    float distG = segMean[1]-tileMean[1];
    float distR = segMean[2]-tileMean[2];

    return sqrt(distB*distB + distG*distG + distR*distR);
}

Mat findBestMeans(const Scalar& segMean, vector<Mat> tiles)
{
    float bestDist = 999999;
    int bestTile = 0; // indexul celui mai bun tile

    for(int i=0;i<tiles.size();i++){
        Scalar tileMean = computeMeans(tiles[i]);
        float dist = compareMeans(segMean, tileMean);
        if (dist<bestDist){
            bestDist=dist;
            bestTile=i;
        }
    }

    return tiles[bestTile];
}

///FUNCTII PENTRU HISTOGRAME
histogramsRGB computeHistograms(Mat img)
{
    vector<float> hR(256,0.0);
    vector<float> hG(256,0.0);
    vector<float> hB(256,0.0);
    int totalPixels = img.rows*img.cols;    //numarul total de pixeli dintr-o imagine

    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++)
        {
                Vec3b pixel = img.at<Vec3b>(i,j);
                hB[pixel[0]]++;
                hG[pixel[1]]++;
                hR[pixel[2]]++;
        }

    //normalizarea histogramelor
    for(float& val:hR)
        val/=float(totalPixels);
    for(float& val:hG)
        val/=float(totalPixels);
    for(float& val:hB)
        val/=float(totalPixels);

    return {hR,hG,hB};
}

float compareHistograms(histogramsRGB h1, histogramsRGB h2)
{
    float distR=0.0,distG=0.0,distB=0.0;

    for (int i=0;i<256;i++) {
        distR+=(h1.hR[i]-h2.hR[i])*(h1.hR[i]-h2.hR[i]);
        distG+=(h1.hG[i]-h2.hG[i])*(h1.hG[i]-h2.hG[i]);
        distB+=(h1.hB[i]-h2.hB[i])*(h1.hB[i]-h2.hB[i]);
    }

    return sqrt(distR + distG + distB);
}

Mat findBestHistograms(const histogramsRGB& segHist, vector<Mat> tiles)
{
    float bestDist = 999999;
    int bestTile = 0;

    for(int i=0;i<tiles.size();i++)
    {
        histogramsRGB tileHist = computeHistograms(tiles[i]);
        float dist = compareHistograms(segHist, tileHist);
        if(dist<bestDist)
        {
            bestDist = dist;
            bestTile = i;
        }
    }

    return tiles[bestTile];
}

///TOP LEVEL PENTRU ALEGEREA TILE-URILOR
vector<Mat> findBestMatches(segments seg, const vector<Mat>& tiles, int op)
{
    vector<Mat> bestTiles;
    if (op==1){ //media de culoare
        for(int i=0;i<seg.s.size();i++){
            Scalar segMeans = computeMeans(seg.s[i]);
            Mat res = findBestMeans(segMeans, tiles);
            bestTiles.push_back(res);
        }
    }
    else if (op==2){ //histograma
        for(int i=0;i<seg.s.size();i++){
            histogramsRGB segHist = computeHistograms(seg.s[i]);
            Mat res = findBestHistograms(segHist, tiles);
            bestTiles.push_back(res);
        }
    }
    else
        throw runtime_error("Error: Optinue gresita! Alegeti 1(media culorilor), 2(histograma)");

    return bestTiles;
}

///ASAMBLARE FINALA
Mat composeMosaic(Mat source, segments seg, vector<Mat> tiles)
{
    Mat result = Mat(source.rows,source.cols, CV_8UC3);
    for(int k = 0;k<tiles.size();k++)
    {
        int x=seg.xs[k];
        int y=seg.ys[k];
        for(int i=0;i<50;i++)
            for(int j=0;j<50;j++)
                result.at<Vec3b>(y+i,x+j) = tiles[k].at<Vec3b>(i,j);
    }

    return result;
}


