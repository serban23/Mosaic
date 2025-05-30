#include <iostream>
#include <opencv2/opencv.hpp>
#include "mozaic.h"
using namespace std;
using namespace cv;

#include <filesystem>
using namespace std::filesystem;

vector<float> bestMeans;
vector<float> bestHistograms;
int segCols=0, segRows=0;

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

    imshow ("Original", source);

    vector<Mat> tiles;

    for (const auto& entry: directory_iterator("C:\\FACULTATE\\3.2\\PI\\PROIECT\\Mozaic\\images\\tiles"))
    {
        Mat img = imread(entry.path().string(), IMREAD_COLOR);
        if (!img.empty())
            tiles.push_back(img);
    }

    return {source,tiles};
}

int processOriginal(Mat &original)
{
    segRows = original.rows / 50;
    segCols = original.cols / 50;

    // redimensionăm imaginea la cel mai apropiat multiplu de 50
    resize(original, original, Size(segCols * 50, segRows * 50));

    return segRows * segCols;
}

///SEGMENTARE
segments imageSegmentation(const Mat& source, int noSegments)
{
    int rows = source.rows;
    int cols = source.cols;

    float gridRows = rows / 50;
    float gridCols = cols / 50;

    if(gridRows * 50 != rows || gridCols*50!=cols)
        throw runtime_error("Eroare: dimensiunile imaginii nu sunt compatibile cu numarul de segmente.");

    segments seg;

    for(int i = 0; i<gridRows; i++)
        for(int j=0;j<gridCols;j++)
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

Scalar computeMeansForTiles(Mat tile)
{
    Vec3b pixel = tile.at<Vec3b>(0,0);
    return Scalar(pixel[2],pixel[1],pixel[0]);
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

    for(int i=0;i<tiles.size();i++)
    {
        Scalar tileMean = computeMeansForTiles(tiles[i]);
        float dist = compareMeans(segMean, tileMean);
        if (dist<bestDist)
        {
            bestDist=dist;
            bestTile=i;
        }
    }

    bestMeans.push_back(bestDist);
    return tiles[bestTile];
}

///FUNCTII PENTRU HISTOGRAME
histogramsRGB computeHistograms(Mat img)
{
    vector<float> hR(256,0.0);
    vector<float> hG(256,0.0);
    vector<float> hB(256,0.0);
    int totalPixels = img.rows*img.cols;

    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++)
        {
                Vec3b pixel = img.at<Vec3b>(i,j);
                hB[pixel[0]]++;
                hG[pixel[1]]++;
                hR[pixel[2]]++;
        }

    if (totalPixels>0)
    {
        for (int i=0; i<256; i++)
        {
            hR[i]/=totalPixels;
            hG[i]/=totalPixels;
            hB[i]/=totalPixels;
        }
    }

    return {hR,hG,hB};
}

histogramsRGB computeHistogramsForTiles(Mat tile)
{
    vector<float> hR(256, 0.0);
    vector<float> hG(256, 0.0);
    vector<float> hB(256, 0.0);

    Vec3b pixel = tile.at<Vec3b>(0, 0);

    hB[pixel[0]] = 1.0f;
    hG[pixel[1]] = 1.0f;
    hR[pixel[2]] = 1.0f;

    return {hR, hG, hB};
}

float compareHistograms(histogramsRGB h1, histogramsRGB h2)
{
    float distR=0.0,distG=0.0,distB=0.0;

    for (int i=0;i<256;i++)
    {
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
        histogramsRGB tileHist = computeHistogramsForTiles(tiles[i]);
        float dist = compareHistograms(segHist, tileHist);
        if(dist<bestDist)
        {
            bestDist = dist;
            bestTile = i;
        }
    }

    //printf("Histogram dist: %lf", bestDist);
    bestHistograms.push_back(bestDist);
    return tiles[bestTile];
}

///FUNCTIE PENTRU AFISAREA DIFERENTELOR INTRE ORIGINAL SI MOZAIC
Mat vectorToMat(vector<float> values)
{
    Mat m(segRows*50,segCols*50,CV_8UC1);

    float minVal = *min_element(values.begin(), values.end());
    float maxVal = *max_element(values.begin(), values.end());

    printf("\nMin-Max: %lf, %lf\n", minVal, maxVal);

    for(int i=0;i<segRows;i++)
        for(int j=0;j<segCols;j++)
        {
            float val = values[i*segCols +j];

            int pixelVal = 0;
            if (maxVal > minVal) {
                pixelVal = static_cast<int>(255.0f * (val - minVal) / (maxVal - minVal));
            }

            Rect tile(j*50,i*50, 50,50);
            rectangle(m,tile,Scalar(pixelVal), FILLED);
        }

    return m;
}

///TOP LEVEL PENTRU ALEGEREA TILE-URILOR
vector<Mat> findBestMatches(segments seg, const vector<Mat>& tiles, int op)
{
    vector<Mat> bestTiles;
    if (op==1)
    { //media de culoare
        for(int i=0;i<seg.s.size();i++)
        {
            Scalar segMeans = computeMeans(seg.s[i]);
            Mat res = findBestMeans(segMeans, tiles);
            bestTiles.push_back(res);
        }
        Mat m = vectorToMat(bestMeans);
        imshow("DifFerence between means", m);
    }
    else if (op==2)
    { //histograma
        for(int i=0;i<seg.s.size();i++)
        {
            histogramsRGB segHist = computeHistograms(seg.s[i]);
            Mat res = findBestHistograms(segHist, tiles);
            bestTiles.push_back(res);
        }
        Mat m = vectorToMat(bestHistograms);
        imshow("DifFerence between histograms", m);
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


