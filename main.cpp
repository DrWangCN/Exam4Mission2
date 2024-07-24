#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

vector<Mat> Src;
Mat mask;
int gray_equalized;

void Process1 (Mat gray, int i);
void Process2 (Mat img, Mat gray, int i);
bool isYellowOrSimilar(const Vec3b& hsvPixel);
bool isWhiteOrSimilar(const Vec3b& hsvPixel);
int LSLimit;

int main() {
    mask = imread("./dataset/car_mask.png", IMREAD_GRAYSCALE);    // 读取掩模图像
    if (mask.empty()) {
        cout << "Could not read the mask!" << endl;
        return 1;
    }
    mask = 255 - mask;

    for (int i = 1; i<=5; i++) {
        string path = "./dataset/" + to_string(i) + ".jpg";
        Mat img = imread(path);
        if (img.empty()) {
            cout << "Could not read the image: " << path << endl;
            return 1;
        }
        Src.push_back(img);         // 读取图像

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);    // 转换为灰度图像
        GaussianBlur(gray, gray, Size(5, 5), 0);    // 高斯滤波
        equalizeHist(gray, gray);    // 直方图均衡化
        createCLAHE(2.0, Size(8, 8))->apply(gray, gray);    // 自适应直方图均衡化
        GaussianBlur(gray, gray, Size(5, 5), 0);    // 高斯滤波
        imwrite("./output/grey/" + to_string(i) + "_grayequalized.jpg", gray);

        Process2(img, gray, i);
    }

    cout << "Done!" << endl;
}

// void Process1 (Mat gray, int i) {
//     Mat gauss;
//         GaussianBlur(gray, gauss, Size(5, 5), 0);    // 高斯滤波

//         float height = gauss.rows;
//         float width = gauss.cols;

//         // 裁剪
//         vector<vector<Rect>> rects;
//         vector<vector<Mat>> binRegion;
//         for (int i = 0; i<8; i++) {
//             vector<Rect> rect;
//             for (int j = 0; j<4; j++) {
//                 rect.push_back(Rect(i * width / 8, j * height / 4, width / 8, height / 4));
//             }
//             rects.push_back(rect);
//         }
        
//         // 二值化
//         for (int i = 0; i<rects.size(); i++) {
//             vector<Mat> tmpbinRow;
//             for (int j = 0; j<rects[i].size(); j++) {
//                 Mat tmpBin;
//                 if (j == 0 || j == 1) {
//                     threshold(gauss(rects[i][j]), tmpBin, 255, 255, THRESH_BINARY_INV);
//                 }
//                 else if (j == 2)
//                 {
//                     if (i < 2 || i > 5) {
//                         threshold(gauss(rects[i][j]), tmpBin, 50, 255, THRESH_BINARY_INV);
//                     }
//                     else {
//                         threshold(gauss(rects[i][j]), tmpBin, 70, 255, THRESH_BINARY_INV);
//                     }
//                 }
//                 else {
//                     if (i < 1 || i > 6) {
//                         threshold(gauss(rects[i][j]), tmpBin, 40, 255, THRESH_BINARY_INV);
//                     }
//                     else{
//                         threshold(gauss(rects[i][j]), tmpBin, 50, 255, THRESH_BINARY_INV);                        
//                     }
//                 }
//                 tmpbinRow.push_back(tmpBin);
//             }
//             binRegion.push_back(tmpbinRow);
//         }

//         Mat binary = Mat::zeros(gray.size(), gray.type());

//         for (int i = 0; i<rects.size(); i++) {
//             for (int j = 0; j<rects[i].size(); j++) {
//                 if ((i == 3 || i == 4) && j == 2) 
//                 {
//                     dilate(binRegion[i][j], binRegion[i][j], Mat(), Point(-1, -1), 1);
//                 }
//                 else {
//                     dilate(binRegion[i][j], binRegion[i][j], Mat(), Point(-1, -1), 2);
//                 }
//                 binRegion[i][j].copyTo(binary(rects[i][j]));
//             }
//         }

//         Mat masked_img;
//         binary = 255 - binary;
//         bitwise_and(binary, mask, masked_img);    // 使用掩模

//         dilate(masked_img, masked_img, Mat(), Point(-1, -1), 1);    // 膨胀
//         erode(masked_img, masked_img, Mat(), Point(-1, -1), 1);     // 腐蚀
//         dilate(masked_img, masked_img, Mat(), Point(-1, -1), 2);    // 膨胀
//         erode(masked_img, masked_img, Mat(), Point(-1, -1), 3);     // 腐蚀
//         dilate(masked_img, masked_img, Mat(), Point(-1, -1), 1);    // 膨胀

//         imwrite("./output/" + to_string(i) + ".jpg", masked_img);

//         // cvtColor(img, img, COLOR_BGR2HSV);
//         // for (int i = 0; i<img.rows; i++) {
//         //     for (int j = 0; j<img.cols; j++) {
//         //         Vec3b& pixel = img.at<Vec3b>(i, j);
//         //         pixel[1] = 255;
//         //         pixel[2] = 255;
//         //     }
//         // }
//         // cvtColor(img, img, COLOR_HSV2BGR);
//         // imwrite("./test/" + to_string(i) + "_hsv.jpg", img);
// }

void Process2 (Mat img, Mat gray, int i) {
    Mat gauss;
    GaussianBlur(gray, gauss, Size(5, 5), 0);    // 高斯滤波

    Mat binary;
    threshold(gauss, binary, 165, 255, THRESH_BINARY_INV);    // 二值化

    Mat masked_img;
    binary = 255 - binary;
    bitwise_and(binary, mask, masked_img);    // 使用掩模

    Mat HLS;
    switch (i)
    {
    case 1: LSLimit = 180; break;
    case 2: case 4: LSLimit = 220; break;
    case 5: LSLimit = 180; break;
    case 3: LSLimit = 320; break;
    default:
        break;
    }
    cvtColor(img, HLS, COLOR_BGR2HLS);
    for (int i = 0; i<masked_img.rows; i++) {
        for (int j = 0; j<masked_img.cols; j++) {
            if (masked_img.at<uchar>(i, j) == 255) {
                Vec3b& pixel = HLS.at<Vec3b>(i, j);
                if ((!isYellowOrSimilar(pixel) && !isWhiteOrSimilar(pixel)) || i < masked_img.rows * 0.4) {
                    uchar& pixelmapping = masked_img.at<uchar>(i, j);
                    pixelmapping = 0;
                }
            }
        }
    }

    erode(masked_img, masked_img, Mat(), Point(-1, -1), 2);
    dilate(masked_img, masked_img, Mat(), Point(-1, -1), 3);
    erode(masked_img, masked_img, Mat(), Point(-1, -1), 3);
    dilate(masked_img, masked_img, Mat(), Point(-1, -1), 2);
    erode(masked_img, masked_img, Mat(), Point(-1, -1), 1);
    dilate(masked_img, masked_img, Mat(), Point(-1, -1), 1);

    imwrite("./output/" + to_string(i) + ".jpg", masked_img);
}

bool isYellowOrSimilar(const Vec3b& hlsPixel) {
    int hue = hlsPixel[0]; 
    int lightness = hlsPixel[1]; 
    int saturation = hlsPixel[2]; 

    bool hueInRange = (hue <= 32 && hue >= 13);
    bool LS = lightness + saturation > LSLimit;

    return hueInRange && LS;
}

bool isWhiteOrSimilar(const Vec3b& hlsPixel) {
    int hue = hlsPixel[0]; 
    int lightness = hlsPixel[1]; 
    int saturation = hlsPixel[2]; 

    bool lightInRange = lightness > 230;

    return lightInRange;
}

