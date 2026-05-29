#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

vector<string> emotions = {
    "Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear"
};

vector<float> softmax(const Mat& scores) {
    vector<float> result;
    float maxVal = -999999;

    for (int i = 0; i < scores.cols; i++) {
        maxVal = max(maxVal, scores.at<float>(0, i));
    }

    float sum = 0.0;
    for (int i = 0; i < scores.cols; i++) {
        float e = exp(scores.at<float>(0, i) - maxVal);
        result.push_back(e);
        sum += e;
    }

    for (float& v : result) v /= sum;
    return result;
}

int main() {
    CascadeClassifier faceCascade;

    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "haarcascade_frontalface_default.xml file paoa jay nai!" << endl;
        return -1;
    }

    dnn::Net emotionNet;

    try {
        emotionNet = dnn::readNetFromONNX("emotion-ferplus-8.onnx");
    } catch (const cv::Exception& e) {
        cout << "emotion-ferplus-8.onnx load hoy nai!" << endl;
        cout << e.what() << endl;
        return -1;
    }

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "Camera open hoy nai!" << endl;
        return -1;
    }

    Mat frame, gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, Size(80, 80));

        for (Rect face : faces) {
            Mat faceROI = gray(face);
            resize(faceROI, faceROI, Size(64, 64));
            faceROI.convertTo(faceROI, CV_32F, 1.0 / 255.0);

            Mat blob = dnn::blobFromImage(faceROI);

            emotionNet.setInput(blob);
            Mat output = emotionNet.forward();

            output = output.reshape(1, 1);

            vector<float> probs = softmax(output);

            int best = 0;
            float bestProb = probs[0];

            for (int i = 1; i < probs.size() && i < emotions.size(); i++) {
                if (probs[i] > bestProb) {
                    bestProb = probs[i];
                    best = i;
                }
            }

            string label = emotions[best] + " " + to_string((int)(bestProb * 100)) + "%";

            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, label, Point(face.x, face.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }

        imshow("Emotion Recognition", frame);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
