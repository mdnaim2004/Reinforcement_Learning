/*
 * Emotion Detection - Fixed C++ Version
 * Uses: Haar Cascade for face detection + FER+ ONNX for emotion recognition
 *
 * Required files in same folder:
 *   haarcascade_frontalface_default.xml
 *   emotion-ferplus-8.onnx
 *
 * Build:
 *   g++ main.cpp -o emotion $(pkg-config --cflags --libs opencv4)
 *
 * Run:
 *   ./emotion
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <unistd.h>

using namespace cv;
using namespace std;

// FER+ model labels
static const vector<string> EMOTIONS = {
    "Neutral", "Happy", "Surprise", "Sad",
    "Angry", "Disgust", "Fear", "Contempt"
};

static const vector<Scalar> EMO_COLORS = {
    Scalar(180,180,180), // Neutral
    Scalar(50,220,50),   // Happy
    Scalar(255,165,0),   // Surprise
    Scalar(210,80,80),   // Sad
    Scalar(50,50,220),   // Angry
    Scalar(0,180,100),   // Disgust
    Scalar(220,100,220), // Fear
    Scalar(120,120,200)  // Contempt
};

static const int SMOOTH_FRAMES = 5;
static const float MIN_CONF = 0.35f;

bool fileExists(const string& path) {
    return access(path.c_str(), F_OK) == 0;
}

string findHaarCascade() {
    const vector<string> paths = {
        "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
    };

    for (const auto& path : paths) {
        if (fileExists(path)) return path;
    }

    return "";
}

vector<float> softmax(Mat scores) {
    scores = scores.reshape(1, 1);

    int n = scores.cols;
    vector<float> result(n);

    const float* ptr = scores.ptr<float>(0);

    float maxVal = *max_element(ptr, ptr + n);
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        result[i] = exp(ptr[i] - maxVal);
        sum += result[i];
    }

    if (sum <= 0.0f) return result;

    for (float& v : result) {
        v /= sum;
    }

    return result;
}

Rect expandBox(const Rect& r, const Size& bounds) {
    int addW = int(r.width * 0.20f);
    int addTop = int(r.height * 0.35f);
    int addBottom = int(r.height * 0.10f);

    int x = max(0, r.x - addW / 2);
    int y = max(0, r.y - addTop);
    int w = min(bounds.width - x, r.width + addW);
    int h = min(bounds.height - y, r.height + addTop + addBottom);

    return Rect(x, y, w, h);
}

void drawBars(Mat& frame, const vector<float>& avg, int bestIdx) {
    if (avg.empty()) return;

    int barW = 100;
    int barH = 12;
    int labelW = 62;
    int panelW = labelW + barW + 42;
    int panelH = (int)avg.size() * 22 + 10;

    int px = max(0, frame.cols - panelW - 8);
    int py = 8;

    Rect panel(px, py, min(panelW, frame.cols - px), min(panelH, frame.rows - py));
    if (panel.width <= 0 || panel.height <= 0) return;

    Mat roi = frame(panel);
    Mat dark(roi.size(), roi.type(), Scalar(20, 20, 20));
    addWeighted(roi, 0.35, dark, 0.65, 0, roi);
    rectangle(frame, panel, Scalar(70, 70, 70), 1);

    for (int i = 0; i < (int)avg.size() && i < (int)EMOTIONS.size(); i++) {
        int y = py + 18 + i * 22;
        if (y + barH >= frame.rows) break;

        Scalar color = (i == bestIdx) ? EMO_COLORS[i] : Scalar(160, 160, 160);

        putText(frame, EMOTIONS[i], Point(px + 4, y + 2),
                FONT_HERSHEY_SIMPLEX, 0.32, color, 1);

        int bx = px + labelW;
        rectangle(frame, Rect(bx, y - 10, barW, barH), Scalar(50, 50, 50), -1);

        int filled = max(0, min(barW, int(avg[i] * barW)));
        if (filled > 0) {
            rectangle(frame, Rect(bx, y - 10, filled, barH), color, -1);
        }

        string pct = to_string((int)(avg[i] * 100)) + "%";
        putText(frame, pct, Point(bx + barW + 4, y + 2),
                FONT_HERSHEY_SIMPLEX, 0.30, Scalar(190, 190, 190), 1);
    }
}

int main() {
    string haarPath = findHaarCascade();

    if (haarPath.empty()) {
        cerr << "[ERROR] haarcascade_frontalface_default.xml not found." << endl;
        cerr << "Run this command:" << endl;
        cerr << "wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" << endl;
        return -1;
    }

    CascadeClassifier faceCascade;
    if (!faceCascade.load(haarPath)) {
        cerr << "[ERROR] Could not load Haar cascade: " << haarPath << endl;
        return -1;
    }

    string modelPath = "emotion-ferplus-8.onnx";
    if (!fileExists(modelPath)) {
        cerr << "[ERROR] emotion-ferplus-8.onnx not found." << endl;
        return -1;
    }

    dnn::Net emotionNet;
    try {
        emotionNet = dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception& e) {
        cerr << "[ERROR] Cannot load ONNX model: " << e.what() << endl;
        return -1;
    }

    if (emotionNet.empty()) {
        cerr << "[ERROR] Emotion model loaded empty." << endl;
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Camera open hoy nai." << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    deque<vector<float>> history;

    Mat frame, gray;
    cout << "[OK] Program started. Press ESC to exit." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1);

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, Size(70, 70));

        if (!faces.empty()) {
            Rect face = *max_element(
                faces.begin(),
                faces.end(),
                [](const Rect& a, const Rect& b) {
                    return a.area() < b.area();
                }
            );

            Rect faceROIBox = expandBox(face, frame.size());

            Mat faceGray = gray(faceROIBox).clone();
            resize(faceGray, faceGray, Size(64, 64));
            faceGray.convertTo(faceGray, CV_32F);

            Mat blob = dnn::blobFromImage(
                faceGray,
                1.0,
                Size(64, 64),
                Scalar(),
                false,
                false,
                CV_32F
            );

            emotionNet.setInput(blob);
            Mat output = emotionNet.forward();

            vector<float> probs = softmax(output);

            if ((int)probs.size() > (int)EMOTIONS.size()) {
                probs.resize(EMOTIONS.size());
            }

            history.push_back(probs);
            if ((int)history.size() > SMOOTH_FRAMES) {
                history.pop_front();
            }

            vector<float> avg(probs.size(), 0.0f);
            for (const auto& p : history) {
                for (int i = 0; i < (int)p.size(); i++) {
                    avg[i] += p[i];
                }
            }

            for (float& v : avg) {
                v /= (float)history.size();
            }

            int bestIdx = 0;
            float bestProb = avg[0];

            for (int i = 1; i < (int)avg.size(); i++) {
                if (avg[i] > bestProb) {
                    bestProb = avg[i];
                    bestIdx = i;
                }
            }

            string label;
            if (bestProb < MIN_CONF) {
                label = "Uncertain " + to_string((int)(bestProb * 100)) + "%";
            } else {
                label = EMOTIONS[bestIdx] + " " + to_string((int)(bestProb * 100)) + "%";
            }

            Scalar color = EMO_COLORS[bestIdx];

            rectangle(frame, face, color, 2);

            int baseLine = 0;
            Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);

            int x = face.x;
            int y = max(25, face.y - 8);

            rectangle(frame,
                      Point(x - 2, y - textSize.height - 5),
                      Point(x + textSize.width + 4, y + baseLine + 2),
                      Scalar(15, 15, 15),
                      -1);

            putText(frame, label, Point(x, y),
                    FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

            drawBars(frame, avg, bestIdx);
        } else {
            history.clear();
            putText(frame, "No face detected", Point(20, 35),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }

        imshow("Emotion Recognition", frame);

        if (waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}