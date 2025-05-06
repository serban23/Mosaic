#ifndef LAB7_H
#define LAB7_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


//structura in care stocam imaginea originala si imaginile mici
typedef struct {
    Mat source;
    vector<Mat> tiles;
}images;

//structura in care stocam segmentul din imaginea originala si x-ul si y-ul coltului din stanga sus
typedef struct{
    vector<Mat> s;
    vector<int> xs;
    vector<int> ys;
}segments;

//structura in care stocam histogramele fiecarui canal de culoare
typedef struct{
    vector<float> hR;
    vector<float> hG;
    vector<float> hB;
}histogramsRGB;

void generateTiles();

images loadImages(const string& originalName);

segments imageSegmentation(const Mat& source, int noSegments);

Scalar computeMeans(Mat img);

float compareMeans(Scalar segMean, Scalar tileMean);

Mat findBestMeans(const Scalar& segMean, vector<Mat> tiles);

histogramsRGB computeHistograms(Mat img);

float compareHistograms(histogramsRGB h1, histogramsRGB h2);

Mat findBestHistograms(const histogramsRGB& segHist, vector<Mat> tiles);

vector<Mat> findBestMatches(segments seg, const vector<Mat>& tiles, int op);

Mat composeMosaic(Mat source, segments seg, vector<Mat> tiles);


#endif