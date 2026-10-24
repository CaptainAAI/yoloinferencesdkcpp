// YoloHttp.h
//
// Public API of YoloHttp: object detection through the YOLO FastAPI server
// (server/server.py: POST /predict, GET /models) over HTTP.
//
// Typical use:
//
//     yolohttp::Client client("140.129.7.180:6767");     // once
//     if (!client.Connect())
//         std::cerr << client.LastError();
//
//     yolohttp::Result r = client.Detect("model1", frame); // per image
//     if (!r.ok)
//         std::cerr << r.error;
//     yolohttp::Draw(frame, r.detections);                 // boxes + labels
//
// Boxes only, without the class names over the image:
//
//     yolohttp::DrawOptions opt;
//     opt.labels = false;
//     yolohttp::Draw(frame, r.detections, opt);
//
// Several models at the same time (each model runs on its own server thread,
// so these requests do not wait for each other):
//
//     auto f1 = client.DetectAsync("model1", frame);
//     auto f2 = client.DetectAsync("model2", frame);
//     auto f3 = client.DetectAsync("model3", frame);
//     yolohttp::Result r1 = f1.get(), r2 = f2.get(), r3 = f3.get();
//
// Notes:
// - A Client is thread safe: Detect()/DetectAsync() may be called from any
//   number of threads at once. Each call returns its own Result, so errors
//   and timings never mix between threads. HTTP connections are kept alive
//   and reused between calls.
// - std::string, std::vector, std::future and cv::Mat cross the DLL boundary,
//   so the calling project must use the same C++ runtime as the DLL: x64, /MD
//   (Multi-threaded DLL), _DEBUG not defined.

#pragma once
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// YOLOHTTP_EXPORTS is defined only by the DLL project (NewSDKhttp.vcxproj).
// Projects that use the DLL do not define it, so the functions are imported.
#ifdef YOLOHTTP_EXPORTS
#define YH_API __declspec(dllexport)
#else
#define YH_API __declspec(dllimport)
#endif

namespace yolohttp {

// One detected object.
struct Detection {
    cv::Rect    box;       // x, y, width, height in pixels of the image passed to Detect()
    float       score;     // confidence, 0.0 - 1.0
    int         class_id;  // class index returned by the model
    std::string label;     // class name from the model, or the class index as text
};

// Where the time of one request went, in milliseconds.
// round_trip and encode are measured here; the others come from the server.
struct Timing {
    double encode         = 0;  // JPEG encoding of the cv::Mat on this PC (0 for encoded bytes)
    double round_trip     = 0;  // request sent -> response received (network + server)
    double total_server   = 0;  // whole request inside the server
    double upload_read    = 0;  // server reading the uploaded file
    double queue_wait     = 0;  // waiting for the model's thread (other requests for the same model)
    double decode         = 0;  // server decoding the JPEG/PNG
    double inference_call = 0;  // model.predict() as a whole
    double preprocess     = 0;  // YOLO stages inside predict()
    double inference      = 0;
    double postprocess    = 0;
    double serialization  = 0;  // server turning boxes into a list
};

// Result of one Detect() call.
struct Result {
    bool                   ok = false;
    std::string            error;        // empty when ok
    int                    http_status = 0;  // 0 = no HTTP response (server unreachable, timeout)
    std::string            model;
    std::vector<Detection> detections;   // boxes in pixels of the original image
    Timing                 timing_ms;
};

// The models the server has loaded, from GET /models.
// `name` is what Detect() takes; `file` is only the engine behind it, so the
// two can differ (e.g. name "omkegas", file "model3.engine").
struct ModelInfo {
    std::string                name;         // pass this to Detect()/DetectAsync()
    std::string                file;         // model file on the server, e.g. "model3.engine"
    std::string                status;       // "ready", "loading" or "error" (Detect works only when "ready")
    std::map<int, std::string> class_names;  // class id -> name
    int                        workers = 0;  // server threads for this model
};

#pragma warning(push)
#pragma warning(disable : 4251)  // std::shared_ptr member: callers use the same runtime anyway

class YH_API Client {
public:
    // url        - "IP:port" of the server, e.g. "140.129.7.180:6767";
    //              "http://" in front is accepted too
    // timeout_ms - limit for a whole request (connect + upload + inference + response)
    // Nothing is sent yet; Connect() is optional.
    explicit Client(const std::string& url, int timeout_ms = 10000);
    ~Client();

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Asks the server which models it has loaded (GET /models).
    // Returns true when the server answered; Models() then holds the list.
    // On failure returns false; LastError() holds the reason.
    // Detect() works without calling this first.
    bool Connect();

    // Models returned by the last successful Connect().
    std::vector<ModelInfo> Models() const;

    // Error message of the last Connect(); empty when it succeeded.
    // Detect() reports its errors in Result::error instead.
    std::string LastError() const;

    const std::string& Url() const;

    // Runs object detection on one image with the given model.
    //   model       - "model1", "model2", ... as the server names them
    //   frame       - BGR image (CV_8UC3, or 1/4 channels) of any size; it is
    //                 JPEG-encoded before sending
    //   conf        - minimum confidence to keep a detection
    //   iou         - IoU threshold for non-maximum suppression
    //   jpegQuality - JPEG quality 0 - 100 (higher = bigger request, closer to the original)
    // Blocks until the server answers. Check Result::ok.
    Result Detect(const std::string& model, const cv::Mat& frame,
                  float conf = 0.25f, float iou = 0.45f, int jpegQuality = 95) const;

    // Same, for an image that is already encoded (bytes of a .jpg/.png/.bmp file):
    // sent as is, with no decoding or re-encoding on this side.
    Result Detect(const std::string& model, const std::vector<uint8_t>& encoded,
                  float conf = 0.25f, float iou = 0.45f) const;

    // Same as Detect(), but returns at once; the request runs on its own
    // thread and the future holds the Result. Start several (different
    // models, or several images) and get() them afterwards.
    // The frame is shared, not copied: do not draw into it or overwrite it
    // until the future is ready (pass frame.clone() if you need to).
    // The Client may be destroyed before the futures are ready.
    std::future<Result> DetectAsync(const std::string& model, const cv::Mat& frame,
                                    float conf = 0.25f, float iou = 0.45f, int jpegQuality = 95) const;
    std::future<Result> DetectAsync(const std::string& model, std::vector<uint8_t> encoded,
                                    float conf = 0.25f, float iou = 0.45f) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;  // shared with running DetectAsync() threads
};

#pragma warning(pop)

// How Draw() marks the detections. The defaults draw every box with a
// "label score" tag on a see-through background.
struct DrawOptions {
    bool   boxes      = true;   // draw the box rectangles
    bool   labels     = true;   // draw a tag with the class name next to each box
    bool   scores     = true;   // add the score to the tag: "Particle 0.58"
    double tag_alpha  = 0.6;    // tag background opacity: 1 = solid, 0 = text only
    int    thickness  = 0;      // box line width in pixels; 0 = from the image width (2 px per 640 px)
    double font_scale = 0;      // tag text size (OpenCV font scale); 0 = from the image width
};

// Draws each detection's box and "label score" tag onto the image. A tag sits
// above its box, or below it when there is no room above, so it does not cover
// the object itself. The image is modified in place; draw on a clone to keep
// the original.
YH_API void Draw(cv::Mat& frame, const std::vector<Detection>& dets);
YH_API void Draw(cv::Mat& frame, const std::vector<Detection>& dets, const DrawOptions& options);

}  // namespace yolohttp
