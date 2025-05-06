#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "src/mozaic.h"
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main() {

    images img;
    segments seg;
    vector<Mat> bestTilesMean, bestTilesHist;

    fs::path tilesPath("C:\\FACULTATE\\3.2\\PI\\PROIECT\\Mozaic\\images\\tiles");

    if (!fs::exists(tilesPath)) {
        fs::create_directory(tilesPath);
        generateTiles();
        std::cout << "Folderul tiles nu exista. L-am creat si am generat tile-urile." << std::endl;
    }
    else if(fs::is_empty(tilesPath)){
        generateTiles();
        cout << "Am generat tile-urile." << endl;
    }
    else{
        cout << "Folderul tiles NU este gol. Nu regeneram." << endl;
    }

    try{
        img = loadImages("leaf");
    }
    catch(runtime_error& e){
        //prindem eroare pentru numele incorect al imaginii mari
        cerr << e.what() << endl;
        return 0;
    }

    try{
        seg = imageSegmentation(img.source, 256);
    }
    catch(runtime_error& e){
        //prindem eroarea de dimensiune
        cerr << e.what() << endl;
        return 0;
    }

    try{
        bestTilesMean = findBestMatches(seg, img.tiles, 1);
        bestTilesHist = findBestMatches(seg, img.tiles, 2);
    }
    catch(runtime_error& e){
        //prindem eroarea pentru optiune invalida
        cerr << e.what() << endl;
        return 0;
    }

    Mat result1 = composeMosaic(img.source, seg, bestTilesMean);
    Mat result2 = composeMosaic(img.source, seg, bestTilesHist);

    imshow("Result1", result1);
    imshow("Result2", result2);

    //for(int i=0;i<bestTiles.size();i++)
        //imshow("result_" + std::to_string(i) + ".bmp", bestTiles[i]);

    waitKey(0);
    return 0;
}
