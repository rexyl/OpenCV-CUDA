#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

Mat src; Mat dst;

int main( int argc, char** argv )
{
    /// Load the source image
    src = imread( 'fruit.jpg', 1 );
    waitKey();
    return 0;
}
