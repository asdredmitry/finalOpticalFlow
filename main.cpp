#include <iostream>
#include <cmath>
#include <ctype.h>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

#include <unistd.h>

using namespace cv;
using namespace std;

const double mu = 0;
const double sigma = 1;

const double eps = 1e-9;
const double epsForSpeed = 0.1;

const int MAX_COUNT = 500;

const int addPoints = 150;
const int critPoints = 150;

//double diff(Point2f & pt, Mat & mat, bool x);
//double diffTime(Point & pt, Mat & cur, Mat & prev);
//double gaussianFunc(double x);
void drawArrow(Mat &mat, Point2f &pt1, Point2f &pt2, Scalar sc);
//double opticalFlowOpenCV(VideoCapture &cap);
//double opticalFlowNaive(VideoCapture &cap);
void calcOpticalFlowWeight(Mat &grayPrev, Mat &grayCur, vector<Point2f> &pointsCur, vector<Point2f> &pointsNext, Size &winSize, Mat &W);
void calcOpticalFlowWeightIter(Mat &grayPrev, Mat &grayCur, vector<Point2f> &pointsCur, vector<Point2f> &vel, Size &winSize, Mat &W, int iter);
//double diff(Point2f & pt, Mat & mat, bool x)
//{
//    return ((double)mat.at<uchar>(pt.y + (int)!x,pt.x + (int)x) - (double)mat.at<uchar>(pt.y - (int)!x,pt.x - (int)x))/2;
//}
//double diffTime(Point2f & pt, Mat & cur, Mat & prev)
//{
//    return ((double)cur.at<uchar>(pt.y, pt.x) - (double)prev.at<uchar>(pt.y,pt.x))/2;
//}
double diff(Point2f & point, Mat & cur,bool x)
{
    //std :: cout << point.x  << " points      " << point.y << std :: endl;
    return ((double)((double)cur.at<uchar>(point.y + (int)!x ,point.x + (int)x) - (double)cur.at<uchar>(point.y - (int)!x,point.x - (int)x)))/2;
}
double diffTime(Point2f & point, Mat & cur, Mat & prev)
{
    return ((double)cur.at<uchar>(point.y,point.x) - (double)prev.at<uchar>(point.y,point.x))/2;
}
double gaussianFunc(double x)
{
    return (1/sqrt(2*sigma*M_PI))*exp(-0.5*pow((x - mu)/sigma,2));
}
void drawArrow(Mat & mat, Point2f & pt1, Point2f & pt2, Scalar sc)
{
    Point2f v = (pt2 - pt1)/norm(pt1 - pt2);
    Point2f end = pt1 + 15*v;
    line(mat, pt1, end, sc, 1.5);
    line(mat, end, Point2f(end.x - 2*(v.x*cos(M_PI/4) + v.y*sin(M_PI/4)), end.y - 2*(-v.x*sin(M_PI/4) + v.y*cos(M_PI/4))), sc, 1.5);
    line(mat, end, Point2f(end.x - 2*(v.x*cos(M_PI/4) - v.y*sin(M_PI/4)), end.y - 2*(v.x*sin(M_PI/4) + v.y*cos(M_PI/4))), sc, 1.5);
}
void translateMat(Mat & mat, double offsetX, double offsetY)
{
    Mat translateMat = (Mat_<double>(2, 3) << 1, 0 , offsetX, 0 , 1, offsetY);
    warpAffine(mat, mat, translateMat, mat.size());
}
void opticalFlowOpenCV(VideoCapture & cap)
{
    TermCriteria termcrit(TermCriteria :: COUNT | TermCriteria :: EPS, 20, 0.001);
    Size subPixWinSize(10, 10);
    Size winSize(10, 10);
    Mat grayCur, grayPrev, frame;

    vector<Point2f> points[2];
    vector<Point2f> pointsTmp;
    vector<uchar> status;
    vector<float> err;

    namedWindow("OpticalFlow");

    while(1)
    {
        cap >> frame;
        if(frame.empty())
        {
            std :: cout << "End of file" << std :: endl;
            break;
        }
        cvtColor(frame,grayCur,COLOR_RGB2GRAY);
        if(points[0].size() < critPoints)
        {
            goodFeaturesToTrack(grayCur,pointsTmp,100,0.01,10,Mat(),3,3,0,0.4);
            for(vector<Point2f> :: iterator i = pointsTmp.begin(); i < pointsTmp.end(); i++)
                points[0].push_back(*i);
        }
        if(grayPrev.empty())
            grayPrev = grayCur.clone();
        calcOpticalFlowPyrLK(grayPrev, grayCur, points[0], points[1], status, err, winSize, 1, termcrit, 10, 0.001);
        int k = 0;
        for(int i = k = 0; i < points[1].size(); i++)
        {
            if(!status[i])
                continue;
            points[1][k++] = points[1][i];
            if(norm(points[1][i] - points[0][i]) < epsForSpeed)
                circle(frame, points[0][i], 2, Scalar(10, 255, 10), 2);
            else
                drawArrow(frame, points[0][i], points[1][i], Scalar(100,255,100));
        }
        points[1].resize(k);
        imshow("OpticalFlow", frame);
        swap(points[1], points[0]);
        swap(grayPrev, grayCur);
        waitKey(25);
    }
}
void opticalFlowNaive(VideoCapture & cap)
{
    Mat frame, grayCur, grayPrev;
    Size winSize(10,10);
    vector<Point2f> curPoints;
    vector<Point2f> velocity;
    Size amountPoints(30,30);
    namedWindow("OpticalFlowNaive");
    bool first = true;

    Mat W = Mat :: zeros(winSize.area(), winSize.area(), CV_64F);
    for(int i = 0; i < winSize.area(); i++)
        W.at<double>(i, i) = 1;
    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std :: cout << "End of file" << std :: endl;
            break ;
        }
        if(first)
        {
            for(int i = 0; i < amountPoints.height; i++)
            {
                for(int j = 0; j < amountPoints.width; j++)
                    curPoints.push_back(Point2f((j + 1)*frame.cols/(amountPoints.width + 1),(i + 1)*frame.rows/(amountPoints.height + 1)));
            }
            first = false;
        }
        cvtColor(frame, grayCur, CV_RGB2GRAY);
        if(grayPrev.empty())
            grayPrev = grayCur.clone();
        velocity = curPoints;
        calcOpticalFlowWeight(grayPrev, grayCur, curPoints, velocity, winSize, W);
        for(int i = 0; i < curPoints.size(); i++)
        {
            Point2f p = curPoints[i] + velocity[i];
            if(norm(velocity[i]) > epsForSpeed)
                drawArrow(frame,curPoints[i], p ,Scalar(0,0,255));
            else
                circle(frame,curPoints[i],2,Scalar(0,0,255));
        }
        imshow("OpticalFlowNaive",frame);
        waitKey(100);
    }
}
void opticalFlowNaiveWeight(VideoCapture & cap)
{
    Mat frame, grayCur, grayPrev;
    Size winSize(10,10);
    vector<Point2f> curPoints;
    vector<Point2f> velocity;
    Size amountPoints(30,30);
    bool first = true;
    namedWindow("OpticalFlowNaiveWeight");
    Mat W = Mat :: zeros(winSize.area(), winSize.area(), CV_64F);
    for(int i = 0; i < winSize.height; i++)
    {
        for(int j = 0; j < winSize.width; j++)
            W.at<double>(i*winSize.width + j,i*winSize.width + j) = gaussianFunc(sqrt(pow(i,2) + pow(j,2)));
    }
    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std :: cout << "End of file" << std :: endl;
            break;
        }
        if(first)
        {
            for(int i = 0; i < amountPoints.height; i++)
            {
                for(int j = 0; j < amountPoints.width; j++)
                    curPoints.push_back(Point2f((j + 1)*frame.cols/(amountPoints.width + 1),(i + 1)*frame.rows/(amountPoints.height + 1)));
            }
            first = false;
        }
        cvtColor(frame, grayCur, CV_RGB2GRAY);
        if(grayPrev.empty())
            grayPrev = grayCur.clone();
        velocity = curPoints;
        calcOpticalFlowWeight(grayPrev, grayCur, curPoints, velocity, winSize, W);
        for(int i = 0; i < curPoints.size(); i++)
        {
            Point2f p = curPoints[i] + velocity[i];
            if(norm(velocity[i]) > epsForSpeed)
                drawArrow(frame, curPoints[i], p, Scalar(0, 0, 255));
            else
                circle(frame, curPoints[i], 2, Scalar(0, 0, 255));
        }
        imshow("OpticalFlowNaiveWeight",frame);
        waitKey(100);
    }
}
void checkOpticalFlow(VideoCapture & cap)
{
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.001);
    Size subPixWinSize(10,10);
    Size winSize(10,10);
    namedWindow("CheckOpticalFlow");
    Mat grayCur, grayPrev, frame;


    vector<Point2f> points[2];
    vector<Point2f> pointsTmp;
    vector<Point2f> velocity;
    vector<uchar> status;
    vector<float> err;

    Mat W = Mat :: zeros(winSize.area(), winSize.area(), CV_64F);
    for(int i = 0; i < winSize.height; i++)
    {
        W.at<double>(i,i) = 1;
    }

    while(true)
    {
        cap >> frame;
        if(frame.empty())
            break;
        cvtColor(frame, grayCur, CV_RGB2GRAY);
        if(grayPrev.empty())
            grayPrev = grayCur.clone();
        if(points[0].size() < critPoints)
        {
            goodFeaturesToTrack(grayCur,pointsTmp, 100, 0.01, 10, Mat(), 3,3,0,0.4);
            cornerSubPix(grayCur, pointsTmp, subPixWinSize, Size(-1,-1), termcrit);
            for(int i = 0; i < pointsTmp.size(); i++)
                points[0].push_back(pointsTmp[i]);
        }
        //for(int i = 0; i < points[0].size(); i++)
        //{
        //        points[0][i].x += (int)(points[0][i].x < 40)*(40) - (int)(points[0][i].x > grayCur.cols - 40)*(40);
        //        points[0][i].y += (int)(points[0][i].y < 40)*(40) - (int)(points[0][i].y > grayCur.rows - 40)*(40);
        //}
        velocity = points[0];
        calcOpticalFlowWeight(grayPrev, grayCur, points[0], velocity, winSize, W);
        calcOpticalFlowPyrLK(grayPrev, grayCur, points[0], points[1], status, err, winSize, 1, termcrit, 10, 0.001);
        int k = 0;
        for(int i = k = 0; i < points[1].size(); i++)
        {
            Point2f p = points[0][i] + velocity[i];
            if(!status[i])
                continue;
            points[1][k++] = points[1][i];
            if(norm(points[1][i] - points[0][i]) < epsForSpeed)
                circle(frame, points[0][i], 2, Scalar(10, 255, 10), 2);
            else
                drawArrow(frame, points[0][i], points[1][i], Scalar(100,255,100));
            if(norm(velocity[i]) > epsForSpeed)
                drawArrow(frame, points[0][i],p, Scalar(0,0,255));
            else
                circle(frame, points[0][i], 2, Scalar(0,0,255));
        }
        points[1].resize(k);
        imshow("CheckOpticalFlow", frame);
        swap(points[1], points[0]);
        swap(grayPrev, grayCur);
        waitKey(25);
    }
}
void calcOpticalFlowWeight(Mat & grayPrev, Mat & grayCur, vector<Point2f> & pointsCur, vector<Point2f> & pointsNext, Size & winSize, Mat & W)
{
    int l = 0;
    for(vector<Point2f> :: iterator cur = pointsCur.begin(); cur < pointsCur.end(); cur++)
    {
        Mat A(winSize.area(), 2, CV_64F);
        Mat b(winSize.area(), 1, CV_64F);
        Mat tmp, B;
        for(int i = 0; i < winSize.height; i++)
        {
            for(int j = 0; j < winSize.width; j++)
            {
                Point2f tpq;
                tpq.y = cur->y + i - winSize.height/2;
                tpq.x = cur->x + j - winSize.width/2;
                A.at<double>(i*winSize.width + j, 1) = diff(tpq, grayCur, 1);
                A.at<double>(i*winSize.width + j, 0) = diff(tpq, grayCur, 0);
                b.at<double>(i*winSize.width + j, 0) = diffTime(tpq, grayCur, grayPrev);
            }
        }
        transpose(A,B);
        tmp = B*W;
        B = tmp.clone();
        tmp = B*A;
        B = tmp.clone();
        if(fabs(determinant(B)) > eps)
        {
            transpose(A,tmp);
            A = tmp.clone();
            tmp = B.inv();
            B = tmp.clone();
            tmp = B*A;
            B = tmp.clone();
            tmp = B*W;
            B = tmp.clone();
            tmp = B*b;
            Point2f pt(tmp.at<double>(0,1), tmp.at<double>(0, 0));
            if(norm(pt) > epsForSpeed)
            {
                pt.x /= -norm(pt);
                pt.y /= -norm(pt);
            }
            else
            {
                pt.x = 0;
                pt.y = 0;
            }
            pointsNext[l] = pt;
        }
        else
        {
            pointsNext[l] = Point2f(0,0);
        }
        l++;
    }
}
void calcOpticalFlowWeightIter(Mat & grayPrev, Mat & grayCur, vector<Point2f> & pointsCur, vector<Point2f> & vel, Size & winSize, Mat & W, int iter)
{

    namedWindow("test");
    Mat tmp1;
    for(int i1 = 0; i1 < pointsCur.size(); i1++)
    {
        for(int j = 0; j < iter; j++)
        {
            if(j == 0)
                vel[i1] = Point2f(0, 0);
            tmp1 = grayCur.clone();
            if(j)
                translateMat(tmp1, vel[i1].x, -vel[i1].y);
            //imshow("test",tmp1);

            Mat A(winSize.area(), 2, CV_64F);
            Mat B, b(winSize.area(), 1, CV_64F);
            Mat tmp;
            for(int i = 0; i < winSize.height; i++)
            {
                for(int j = 0; j < winSize.width; j++)
                {
                    Point2f tpq;
                    tpq.y = pointsCur[i].y + i - winSize.height/2;
                    tpq.x = pointsCur[i].x + j - winSize.width/2;
                    A.at<double>(i*winSize.width + j, 1) = diff(tpq, tmp1, 1);
                    A.at<double>(i*winSize.width + j, 0) = diff(tpq, tmp1, 0);
                    b.at<double>(i*winSize.width + j, 0) = diffTime(tpq, tmp1, grayPrev);
                }
            }
            transpose(A,B);
            tmp = B*W;
            B = tmp.clone();
            tmp = B*A;
            B = tmp.clone();
            if(fabs(determinant(B)) > eps)
            {
                transpose(A,tmp);
                A = tmp.clone();
                tmp = B.inv();
                B = tmp.clone();
                tmp = B*A;
                B = tmp.clone();
                tmp = B*W;
                B = tmp.clone();
                tmp = B*b;
                Point2f pt(tmp.at<double>(0,1), tmp.at<double>(0, 0));
                pt.x *= -1;
                pt.y *= -1;
                vel[i1] += pt;
            }
            if(i1 == 0)
                std :: cout << vel[i1] << std :: endl;
            if(norm(vel[i1]) > epsForSpeed)
            {
                vel[i1].x /= norm(vel[i1]);
                vel[i1].y /= norm(vel[i1]);
            }
        }
    }
}
int main(int argv, char ** argc)
{
    VideoCapture cap;
    if(argv > 2)
    {
        std :: cout << "Unknown input" << std :: endl;
        return 0;
    }
    else if(argv == 2)
    {
        cap.open(argc[1]);
    }
    else if(argv == 1)
    {
        cap.open(0);
    }
    if(!cap.isOpened())
    {
        std :: cout << "Cannot open video flow" << std :: endl;
        return 0;
    }
    //opticalFlowOpenCV(cap);
    //opticalFlowNaiveWeight(cap);
    checkOpticalFlow(cap);
}
