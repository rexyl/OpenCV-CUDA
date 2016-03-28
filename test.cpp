#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat src; Mat dst;

int main( int argc, char** argv )
{
    /// Load the source image
    src = imread( "fruits.jpg", 1 );
    int cols = src.cols, rows = src.rows;
    cout<<"row_num is "<<rows<<endl;
    cout<<"col_num is "<<cols<<endl;
    for(int i = 0; i < rows;i++){
        for(int j=0;j<cols;j++){
            cout<<int(src.data[i*src.cols+j]);
        }
        cout<<endl;
    }
    waitKey();
    return 0;
}