#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <filesystem>
#include <set>

using namespace cv;
using namespace cv::face;
using namespace std;
namespace fs = filesystem;

map<int, string> labelMap;

void loadKnownFaces(
    const string& folderPath,
    vector<Mat>& images,
    vector<int>& labels,
    map<int, string>& labelMap)
{
    int label = 0;

    for (auto& entry : fs::directory_iterator(folderPath))
    {
        string filePath = entry.path().string();
        string fileName = entry.path().stem().string();

        Mat img = imread(filePath, IMREAD_GRAYSCALE);

        if (img.empty()) {
            cout << "Warning: Could not load " << filePath << endl;
            continue;
        }

        Mat resized;
        resize(img, resized, Size(200, 200));

        images.push_back(resized);
        labels.push_back(label);
        labelMap[label] = fileName;

        cout << "Loaded: " << fileName
             << " with label: " << label << endl;
        label++;
    }
}

int main()
{
    //  Load Faces 
    vector<Mat> images;
    vector<int> labels;
    set<string> attendanceLogged;

    cout << "Loading known faces..." << endl;
    loadKnownFaces("knownFaces", images, labels, labelMap);

    if (images.empty()) {
        cout << "Error: No faces loaded!" << endl;
        return -1;
    }

    cout << "Total faces loaded: " << images.size() << endl;

    //  Train Recognizer 
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->train(images, labels);
    cout << "Recognizer trained successfully!" << endl;

    //  Open Webcam 
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Cannot open webcam!" << endl;
        return -1;
    }
    cout << "Webcam opened successfully!" << endl;

    //  Load Face Detector 
    CascadeClassifier faceDetector;
    faceDetector.load("haarcascade_frontalface_default.xml");
    if (faceDetector.empty()) {
        cout << "Error: Cannot load classifier!" << endl;
        return -1;
    }
    cout << "Face detector loaded!" << endl;

    Mat frame;

    //  Main While Loop 
    while (true)
    {
        cap >> frame;

        if (frame.empty()) {
            cout << "Error: Empty frame!" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect Faces 
        vector<Rect> faces;
        faceDetector.detectMultiScale(
            gray, faces, 1.1, 5, 0, Size(30, 30));

        //  Process Each Face 
        for (auto& face : faces)
        {
            Mat faceCrop = gray(face);
            Mat faceResized;
            resize(faceCrop, faceResized, Size(200, 200));

            rectangle(frame, face, Scalar(0, 255, 0), 2);

            //  Recognize 
            int predictedLabel = -1;
            double confidence = 0.0;
            recognizer->predict(faceResized,
                                predictedLabel,
                                confidence);

            string name = "Unknown";
            if (confidence < 100.0) {
                name = labelMap[predictedLabel];
            }

            putText(frame, name,
                    Point(face.x, face.y - 10),
                    FONT_HERSHEY_SIMPLEX,
                    0.8, Scalar(0, 255, 0), 2);

            cout << "Detected: " << name
                 << " | Confidence: " << confidence << endl;

            //  Mark Attendance
           if (name != "Unknown" && attendanceLogged.find(name) == attendanceLogged.end())
                {
                attendanceLogged.insert(name); 

                time_t now = time(0);
                char* dt = ctime(&now);
                string timeStr(dt);
                timeStr.erase(timeStr.end() - 1);

                ofstream file("attendance.csv", ios::app);
                if (file.is_open()) {
                    file << name << "," << timeStr << "\n";
                    file.close();
                    cout << "Attendance logged: "
                        << name << " at " << timeStr << endl;
                }
            }
        }

        imshow("Face Recognition", frame);

        if (waitKey(1) == 'q') {
            cout << "Quitting..." << endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
