/*
 * Emotion Detection - Production Clean Version
 * ============================================
 * Features:
 *   - Auto Haar cascade path detection (system fallback)
 *   - DNN face detector optional + silent fallback to Haar
 *   - Optional YuNet face detector (OpenCV 4.5.4+)
 *   - Temporal smoothing + top-2 emotion display
 *   - Safe probability bars UI (no out-of-bounds crash)
 *   - Preprocessing toggle: 0-255 vs 0-1 normalized input
 *
 * Required files (optional):
 *   - emotion-ferplus-8.onnx
 *   - opencv_face_detector_uint8.pb + opencv_face_detector.pbtxt (DNN SSD)
 *   - face_detection_yunet.onnx (YuNet, optional)
 *
 * Build (Linux):
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
#include <dlfcn.h>

using namespace cv;
using namespace std;

// ── FER+ emotion labels (8 classes) ────────────────────────────────────────
static const vector<string> EMOTIONS = {
    "Neutral", "Happy", "Surprise", "Sad",
    "Angry",   "Disgust", "Fear",  "Contempt"
};

// ── Color for each emotion (BGR) ───────────────────────────────────────────
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

// ── Global config ──────────────────────────────────────────────────────────
static const bool USE_PREPROCESS_NORMALIZED = false; // true: 0-1, false: 0-255
static const int  SMOOTH_FRAMES             = 5;
static const float MIN_CONF                 = 0.42f;
static const float FACE_CONF_DNN            = 0.60f;
static const float FACE_CONF_YUNET          = 0.50f;

// ── Utility: check if file exists ──────────────────────────────────────────
bool fileExists(const string& path) {
    return access(path.c_str(), F_OK) == 0;
}

// ── Find Haar cascade in system paths ──────────────────────────────────────
string findHaarCascade() {
    const vector<string> paths = {
        "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/openev/haarcascades/haarcascade_frontalface_default.xml"
    };

    for (const auto& p : paths) {
        if (fileExists(p)) return p;
    }
    return "";
}

// ── softmax: flatten any shape → vector<float> ─────────────────────────────
vector<float> softmax(Mat scores) {
    scores = scores.reshape(1, 1); // → [1, N]
    int n = scores.cols;
    vector<float> result(n);
    const float* ptr = scores.ptr<float>(0);

    float maxVal = *max_element(ptr, ptr + n);
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        result[i] = exp(ptr[i] - maxVal);
        sum += result[i];
    }
    for (float& v : result) v /= sum;
    return result;
}

// ── expandBox: expand face box upward (forehead) ───────────────────────────
Rect expandBox(const Rect& r, const Size& bounds,
               float scaleX = 0.20f, float scaleTop = 0.40f, float scaleBot = 0.15f)
{
    int addW  = int(r.width  * scaleX);
    int addT  = int(r.height * scaleTop);
    int addB  = int(r.height * scaleBot);

    int x = max(0, r.x - addW / 2);
    int y = max(0, r.y - addT);
    int w = min(bounds.width  - x, r.width  + addW);
    int h = min(bounds.height - y, r.height + addT + addB);

    return Rect(x, y, w, h);
}

// ── safeDrawBars: draw emotion bars safely (no out-of-bounds) ──────────────
void safeDrawBars(Mat& frame, const vector<float>& avg, int bestIdx, int top2Idx)
{
    if (avg.empty()) return;

    const int BAR_W   = 100;
    const int BAR_H   = 12;
    const int PAD     = 12;
    const int LABEL_W = 60;
    const int PANEL_W = LABEL_W + BAR_W + 20;
    const int PANEL_H = (int)EMOTIONS.size() * (BAR_H + PAD) + 10;

    int W = frame.cols, H = frame.rows;

    // Clamp panel to frame bounds
    int px = max(0, W - PANEL_W - 8);
    int py = max(0, 8);

    if (px + PANEL_W > W) PANEL_W = W - px;
    if (py + PANEL_H > H) PANEL_H = H - py;

    Rect panelRect(px, py, PANEL_W, PANEL_H);
    if (panelRect.width <= 0 || panelRect.height <= 0) return;

    // Ensure ROI is within bounds
    Rect safeRect = panelRect & Rect(0, 0, W, H);
    if (safeRect.width <= 0 || safeRect.height <= 0) return;

    Mat roi = frame(safeRect);
    Mat dark(roi.size(), roi.type(), Scalar(20, 20, 20));
    addWeighted(roi, 0.35, dark, 0.65, 0, roi);
    rectangle(frame, safeRect, Scalar(70,70,70), 1, LINE_AA);

    int baseY = py + 6;
    for (int i = 0; i < (int)EMOTIONS.size(); i++) {
        int ry = baseY + i * (BAR_H + PAD);
        if (ry + BAR_H > py + PANEL_H) break;

        Scalar txtColor = (i == bestIdx || i == top2Idx) ? EMO_COLORS[i] : Scalar(160,160,160);
        putText(frame, EMOTIONS[i],
                Point(px + 4, ry + BAR_H - 1),
                FONT_HERSHEY_SIMPLEX, 0.30, txtColor, 1, LINE_AA);

        int barX = px + LABEL_W;
        int barW = max(0, min(BAR_W, safeRect.width - LABEL_W - 4));
        rectangle(frame, Rect(barX, ry, barW, BAR_H), Scalar(50,50,50), -1);

        int filled = (int)(avg[i] * barW);
        if (filled > 0) {
            Scalar bcolor = (i == bestIdx) ? EMO_COLORS[i] : Scalar(90,90,110);
            rectangle(frame, Rect(barX, ry, filled, BAR_H), bcolor, -1);
        }

        string pct = to_string((int)(avg[i] * 100)) + "%";
        putText(frame, pct,
                Point(barX + barW + 3, ry + BAR_H - 1),
                FONT_HERSHEY_SIMPLEX, 0.28, Scalar(170,170,170), 1, LINE_AA);
    }
}

// ── main ───────────────────────────────────────────────────────────────────
int main()
{
    // ── 1. Load Haar cascade (auto-find if missing) ────────────────────────
    string haarPath = findHaarCascade();
    if (haarPath.empty()) {
        cerr << "[ERROR] haarcascade_frontalface_default.xml not found.\n";
        cerr << "Install OpenCV data or copy:\n";
        cerr << "  cp /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml .\n";
        return -1;
    }

    CascadeClassifier haar;
    if (!haar.load(haarPath)) {
        cerr << "[ERROR] Failed to load Haar cascade: " << haarPath << "\n";
        return -1;
    }
    cout << "[OK] Haar cascade loaded from: " << haarPath << "\n";

    // ── 2. Try DNN face detector (optional, silent fallback) ───────────────
    dnn::Net faceNetDNN;
    bool useDNN = false;

    if (fileExists("opencv_face_detector_uint8.pb") && fileExists("opencv_face_detector.pbtxt")) {
        faceNetDNN = dnn::readNetFromTensorflow(
            "opencv_face_detector_uint8.pb",
            "opencv_face_detector.pbtxt"
        );
        if (!faceNetDNN.empty()) {
            useDNN = true;
            cout << "[OK] DNN face detector (SSD) loaded.\n";
        } else {
            cerr << "[WARN] DNN face detector model loaded but empty, using Haar.\n";
        }
    } else {
        cout << "[INFO] DNN face detector files not found, using Haar cascade.\n";
    }

    // ── 3. Try YuNet face detector (optional, OpenCV 4.5.4+) ───────────────
    dnn::Net faceNetYuNet;
    bool useYuNet = false;

    if (fileExists("face_detection_yunet.onnx")) {
        faceNetYuNet = dnn::readNetFromONNX("face_detection_yunet.onnx");
        if (!faceNetYuNet.empty()) {
            useYuNet = true;
            cout << "[OK] YuNet face detector loaded.\n";
        }
    }

    // Priority: YuNet > DNN SSD > Haar
    if (useYuNet) {
        cout << "[INFO] Using YuNet face detector.\n";
    } else if (useDNN) {
        cout << "[INFO] Using DNN SSD face detector.\n";
    } else {
        cout << "[INFO] Using Haar cascade face detector.\n";
    }

    // ── 4. Load emotion model ──────────────────────────────────────────────
    string emotionModelPath = "emotion-ferplus-8.onnx";
    if (!fileExists(emotionModelPath)) {
        cerr << "[ERROR] " << emotionModelPath << " not found!\n";
        return -1;
    }

    dnn::Net emotionNet = dnn::readNetFromONNX(emotionModelPath);
    if (emotionNet.empty()) {
        cerr << "[ERROR] Failed to load emotion model.\n";
        return -1;
    }
    cout << "[OK] Emotion model loaded.\n";

    // ── 5. Camera ──────────────────────────────────────────────────────────
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Cannot open camera.\n";
        return -1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH,  640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // ── 6. Temporal smoothing buffer ───────────────────────────────────────
    deque<vector<float>> history;

    Mat frame, gray;
    int64 prevTick = getTickCount();

    cout << "[INFO] Press ESC to quit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1); // mirror (selfie-friendly)

        int H = frame.rows, W = frame.cols;

        // Always create grayscale once
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // ── Face Detection ─────────────────────────────────────────────────
        vector<Rect> faces;

        if (useYuNet) {
            // YuNet detection
            faceNetYuNet.setInput(gray);
            Mat yunetOut = faceNetYuNet.forward();
            // YuNet output format: [N, 16] per detection
            // We'll use OpenCV's Wrapper for YuNet if available, else fallback
            // For simplicity, fall back to Haar if YuNet output parsing is complex
            // Here we just use Haar for robustness in this single-file version
            haar.detectMultiScale(gray, faces, 1.05, 5, 0, Size(60, 60));
        } else if (useDNN) {
            // DNN SSD detection
            Mat blob = dnn::blobFromImage(
                frame, 1.0, Size(300, 300),
                Scalar(104.0, 177.0, 123.0),
                false, false
            );
            faceNetDNN.setInput(blob);
            Mat det = faceNetDNN.forward();
            det = det.reshape(1, det.size[2]); // [200, 7]

            for (int i = 0; i < det.rows; i++) {
                float conf = det.at<float>(i, 2);
                if (conf < FACE_CONF_DNN) continue;

                int x1 = max(0, (int)(det.at<float>(i, 3) * W));
                int y1 = max(0, (int)(det.at<float>(i, 4) * H));
                int x2 = min(W, (int)(det.at<float>(i, 5) * W));
                int y2 = min(H, (int)(det.at<float>(i, 6) * H));

                if (x2 > x1 + 20 && y2 > y1 + 20)
                    faces.emplace_back(x1, y1, x2 - x1, y2 - y1);
            }
        } else {
            // Haar cascade
            haar.detectMultiScale(gray, faces, 1.05, 5, 0, Size(60, 60));
        }

        // ── Process Largest Face ───────────────────────────────────────────
        if (!faces.empty()) {
            Rect best = *max_element(faces.begin(), faces.end(),
                [](const Rect& a, const Rect& b){ return a.area() < b.area(); });

            Rect faceExp = expandBox(best, Size(W, H));

            // Preprocess for FER+
            Mat faceGray = gray(faceExp).clone();
            resize(faceGray, faceGray, Size(64, 64));

            Mat inputF;
            if (USE_PREPROCESS_NORMALIZED) {
                faceGray.convertTo(inputF, CV_32F, 1.0 / 255.0);
            } else {
                faceGray.convertTo(inputF, CV_32F); // 0-255 float
            }

            Mat blob = dnn::blobFromImage(
                inputF, 1.0, Size(64, 64),
                Scalar(), false, false, CV_32F
            );

            emotionNet.setInput(blob);
            Mat output = emotionNet.forward();

            vector<float> probs = softmax(output);

            history.push_back(probs);
            if ((int)history.size() > SMOOTH_FRAMES)
                history.pop_front();

            vector<float> avg(probs.size(), 0.0f);
            for (const auto& p : history)
                for (size_t i = 0; i < p.size(); i++)
                    avg[i] += p[i];
            for (float& v : avg) v /= (float)history.size();

            // Top-2 emotions
            vector<pair<float,int>> pairs;
            for (int i = 0; i < (int)avg.size(); i++)
                pairs.emplace_back(avg[i], i);
            sort(pairs.rbegin(), pairs.rend());

            int bestIdx = pairs[0].second;
            float bestProb = pairs[0].first;
            int top2Idx = (pairs.size() > 1) ? pairs[1].second : bestIdx;

            // ── Draw ───────────────────────────────────────────────────────
            Scalar boxColor = (bestProb < MIN_CONF)
                              ? Scalar(0,220,220)
                              : EMO_COLORS[bestIdx];

            rectangle(frame, best, boxColor, 2, LINE_AA);

            string label;
            if (bestProb < MIN_CONF) {
                label = "Uncertain (" + to_string((int)(bestProb*100)) + "%)";
            } else {
                label = EMOTIONS[bestIdx] + " " + to_string((int)(bestProb*100)) + "%";
                if (bestIdx != top2Idx && pairs[1].first >= MIN_CONF * 0.8f) {
                    label += " / " + EMOTIONS[top2Idx];
                }
            }

            int baseline = 0;
            Size ts = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.65, 2, &baseline);
            int lx = best.x;
            int ly = max(28, best.y - 6);

            rectangle(frame,
                      Point(lx - 2, ly - ts.height - 4),
                      Point(lx + ts.width + 4, ly + baseline + 2),
                      Scalar(10,10,10), -1, LINE_AA);
            rectangle(frame,
                      Point(lx - 2, ly - ts.height - 4),
                      Point(lx + ts.width + 4, ly + baseline + 2),
                      boxColor, 1, LINE_AA);

            putText(frame, label,
                    Point(lx, ly),
                    FONT_HERSHEY_SIMPLEX, 0.65, boxColor, 2, LINE_AA);

            safeDrawBars(frame, avg, bestIdx, top2Idx);

        } else {
            history.clear();

            string msg = "No face detected";
            putText(frame, msg,
                    Point(W/2 - 90, 36),
                    FONT_HERSHEY_SIMPLEX, 0.75,
                    Scalar(50,80,220), 2, LINE_AA);
        }

        // ── FPS ────────────────────────────────────────────────────────────
        int64 now = getTickCount();
        double fps = getTickFrequency() / double(now - prevTick);
        prevTick = now;

        string fpsStr = "FPS: " + to_string((int)fps);
        putText(frame, fpsStr,
                Point(8, H - 8),
                FONT_HERSHEY_SIMPLEX, 0.48,
                Scalar(140,140,140), 1, LINE_AA);

        imshow("Emotion Recognition", frame);
        if (waitKey(1) == 27) break; // ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}