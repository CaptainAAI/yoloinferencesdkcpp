// YoloHttp.cpp
//
// Implementation of the YoloHttp API (see YoloHttp.h).
// Per image: JPEG-encode (cv::Mat only) -> multipart POST /predict with
// libcurl -> parse the JSON answer with RapidJSON -> Detection list.
// The server already returns final boxes in original image pixels (it runs
// YOLO's own pre- and post-processing), so no NMS or box mapping is done here.
//
// Thread safety: every request takes its own curl easy handle from a small
// pool and returns it afterwards, so any number of threads can send requests
// at once while finished handles keep their HTTP connection alive for the
// next request.

#include "YoloHttp.h"

#include <curl/curl.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rapidjson/document.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <mutex>

namespace yolohttp {

using Clock = std::chrono::steady_clock;

static double MsSince(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ===== HTTP =====

// File part of a POST /predict request.
struct Upload {
    std::string           model;
    const uint8_t*        data = nullptr;
    size_t                size = 0;
    float                 conf = 0;
    float                 iou  = 0;
};

struct Reply {
    long        status = 0;     // HTTP status, 0 when no response arrived
    std::string body;
    std::string error;          // transport error (connection refused, timeout, ...)
    double      ms = 0;         // request sent -> response received
};

struct Client::Impl {
    std::string url;            // "http://host:port", no trailing slash
    long        timeout_ms;

    std::mutex             mu;  // guards everything below
    std::vector<CURL*>     idle;
    std::vector<ModelInfo> models;
    std::string            error;

    Impl(const std::string& u, long timeout) : url(u), timeout_ms(timeout)
    {
        // curl_global_init is not thread safe: run it exactly once.
        static std::once_flag once;
        std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    }

    ~Impl()
    {
        for (CURL* h : idle)
            curl_easy_cleanup(h);
    }

    CURL* Take()
    {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!idle.empty()) {
                CURL* h = idle.back();
                idle.pop_back();
                return h;
            }
        }
        return curl_easy_init();
    }

    void Give(CURL* h)
    {
        curl_easy_reset(h);     // drops the options, keeps the open connection
        std::lock_guard<std::mutex> lock(mu);
        idle.push_back(h);
    }

    Reply  Send(const std::string& path, const Upload* upload);
    Result DetectEncoded(const std::string& model, const uint8_t* data, size_t size, float conf, float iou);
    Result DetectFrame(const std::string& model, const cv::Mat& frame, float conf, float iou, int jpegQuality);
};

static size_t AppendBody(char* data, size_t size, size_t n, void* out)
{
    static_cast<std::string*>(out)->append(data, size * n);
    return size * n;
}

static std::string FormatFloat(float v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

static bool IsPng(const uint8_t* d, size_t n)
{
    return n >= 4 && d[0] == 0x89 && d[1] == 'P' && d[2] == 'N' && d[3] == 'G';
}

// GET path, or POST path as multipart/form-data when upload is set.
Reply Client::Impl::Send(const std::string& path, const Upload* upload)
{
    Reply reply;
    CURL* h = Take();
    if (!h) {
        reply.error = "curl_easy_init failed";
        return reply;
    }

    char errbuf[CURL_ERROR_SIZE] = {};
    const std::string full = url + path;
    curl_easy_setopt(h, CURLOPT_URL, full.c_str());
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, AppendBody);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &reply.body);
    curl_easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);
    curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, std::min(timeout_ms, 5000L));
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);      // required when several threads use curl
    curl_easy_setopt(h, CURLOPT_PROXY, "");         // LAN server: ignore http_proxy settings

    curl_mime*  mime    = nullptr;
    curl_slist* headers = nullptr;
    if (upload) {
        mime = curl_mime_init(h);
        auto field = [&](const char* name, const std::string& value) {
            curl_mimepart* part = curl_mime_addpart(mime);
            curl_mime_name(part, name);
            curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
        };
        field("model_name", upload->model);
        field("conf", FormatFloat(upload->conf));
        field("iou", FormatFloat(upload->iou));

        const bool png = IsPng(upload->data, upload->size);
        curl_mimepart* file = curl_mime_addpart(mime);
        curl_mime_name(file, "file");
        curl_mime_data(file, reinterpret_cast<const char*>(upload->data), upload->size);
        curl_mime_filename(file, png ? "image.png" : "image.jpg");
        curl_mime_type(file, png ? "image/png" : "image/jpeg");
        curl_easy_setopt(h, CURLOPT_MIMEPOST, mime);

        // Without this, curl waits for "100 Continue" before sending bodies over 1 MB.
        headers = curl_slist_append(headers, "Expect:");
        curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    }

    const Clock::time_point t0 = Clock::now();
    const CURLcode rc = curl_easy_perform(h);
    reply.ms = MsSince(t0);

    if (rc == CURLE_OK)
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &reply.status);
    else
        reply.error = errbuf[0] ? errbuf : curl_easy_strerror(rc);

    curl_mime_free(mime);
    curl_slist_free_all(headers);
    Give(h);
    return reply;
}

// ===== JSON =====

static double Num(const rapidjson::Value& obj, const char* key)
{
    if (!obj.IsObject())
        return 0;
    auto it = obj.FindMember(key);
    return it != obj.MemberEnd() && it->value.IsNumber() ? it->value.GetDouble() : 0;
}

// {"0": "person", "1": "car"} -> {0: "person", 1: "car"}
static std::map<int, std::string> ParseNames(const rapidjson::Value& v)
{
    std::map<int, std::string> names;
    if (!v.IsObject())
        return names;
    for (auto it = v.MemberBegin(); it != v.MemberEnd(); ++it)
        if (it->value.IsString())
            names[std::atoi(it->name.GetString())] = it->value.GetString();
    return names;
}

// Error text for a failed request: FastAPI puts it in {"detail": "..."}.
static std::string ErrorText(const Reply& reply)
{
    if (reply.status == 0)
        return reply.error;
    std::string detail = reply.body.substr(0, 300);
    rapidjson::Document doc;
    doc.Parse(reply.body.c_str(), reply.body.size());
    if (!doc.HasParseError() && doc.IsObject()) {
        auto it = doc.FindMember("detail");
        if (it != doc.MemberEnd() && it->value.IsString())
            detail = it->value.GetString();
    }
    return "HTTP " + std::to_string(reply.status) + ": " + detail;
}

// ===== Detection =====

Result Client::Impl::DetectEncoded(const std::string& model, const uint8_t* data, size_t size, float conf, float iou)
{
    Result r;
    r.model = model;
    if (size == 0) {
        r.error = "empty image";
        return r;
    }

    Upload up;
    up.model = model;
    up.data  = data;
    up.size  = size;
    up.conf  = conf;
    up.iou   = iou;
    const Reply reply = Send("/predict", &up);
    r.http_status          = static_cast<int>(reply.status);
    r.timing_ms.round_trip = reply.ms;
    if (reply.status != 200) {
        r.error = ErrorText(reply);
        return r;
    }

    // {"model": ..., "timing_ms": {...}, "names": {"0": ...},
    //  "detections": [[x1, y1, x2, y2, conf, cls], ...]}
    rapidjson::Document doc;
    doc.Parse(reply.body.c_str(), reply.body.size());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("detections")
        || !doc["detections"].IsArray()) {
        r.error = "unexpected response: " + reply.body.substr(0, 300);
        return r;
    }

    if (doc.HasMember("timing_ms")) {
        const rapidjson::Value& t = doc["timing_ms"];
        Timing& tm        = r.timing_ms;
        tm.total_server   = Num(t, "total_server");
        tm.upload_read    = Num(t, "upload_read");
        tm.queue_wait     = Num(t, "queue_wait");
        tm.decode         = Num(t, "decode");
        tm.inference_call = Num(t, "inference_call");
        tm.serialization  = Num(t, "serialization");
        if (t.IsObject() && t.HasMember("model_stages")) {
            const rapidjson::Value& s = t["model_stages"];
            tm.preprocess  = Num(s, "preprocess");
            tm.inference   = Num(s, "inference");
            tm.postprocess = Num(s, "postprocess");
        }
    }

    const std::map<int, std::string> names =
        doc.HasMember("names") ? ParseNames(doc["names"]) : std::map<int, std::string>{};

    for (const rapidjson::Value& row : doc["detections"].GetArray()) {
        if (!row.IsArray() || row.Size() < 6)
            continue;
        float v[6];
        for (rapidjson::SizeType i = 0; i < 6; ++i)
            v[i] = row[i].IsNumber() ? row[i].GetFloat() : 0.f;

        Detection d;
        d.box      = cv::Rect(cv::Point(cvRound(v[0]), cvRound(v[1])),
                              cv::Point(cvRound(v[2]), cvRound(v[3])));
        d.score    = v[4];
        d.class_id = static_cast<int>(v[5]);
        auto it    = names.find(d.class_id);
        d.label    = it != names.end() ? it->second : std::to_string(d.class_id);
        r.detections.push_back(std::move(d));
    }
    r.ok = true;
    return r;
}

Result Client::Impl::DetectFrame(const std::string& model, const cv::Mat& frame, float conf, float iou, int jpegQuality)
{
    if (frame.empty()) {
        Result r;
        r.model = model;
        r.error = "empty image";
        return r;
    }

    const Clock::time_point t0 = Clock::now();
    std::vector<uint8_t> jpg;
    try {
        const int q = std::clamp(jpegQuality, 0, 100);
        if (!cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, q}))
            jpg.clear();
    } catch (const cv::Exception& e) {
        Result r;
        r.model = model;
        r.error = std::string("JPEG encoding failed: ") + e.what();
        return r;
    }
    const double encode_ms = MsSince(t0);

    Result r = DetectEncoded(model, jpg.data(), jpg.size(), conf, iou);
    r.timing_ms.encode = encode_ms;
    if (jpg.empty())
        r.error = "JPEG encoding failed";
    return r;
}

// ===== Client =====

// "140.129.7.180:6767/" -> "http://140.129.7.180:6767"
static std::string NormalizeUrl(std::string url)
{
    const char* space = " \t\r\n";
    url.erase(0, url.find_first_not_of(space));
    url.erase(url.find_last_not_of(space) + 1);
    while (!url.empty() && url.back() == '/')
        url.pop_back();
    if (url.find("://") == std::string::npos)
        url = "http://" + url;
    return url;
}

Client::Client(const std::string& url, int timeout_ms)
    : impl_(std::make_shared<Impl>(NormalizeUrl(url), static_cast<long>(std::max(timeout_ms, 1))))
{
}

Client::~Client() = default;

const std::string& Client::Url() const { return impl_->url; }

bool Client::Connect()
{
    const Reply reply = impl_->Send("/models", nullptr);
    std::string error;
    std::vector<ModelInfo> models;

    if (reply.status != 200) {
        error = ErrorText(reply);
    } else {
        // {"models": {"model1": {"names": {"0": ...}, "workers": 1,
        //             "status": "ready", "file": "model1.engine"}, ...}}
        rapidjson::Document doc;
        doc.Parse(reply.body.c_str(), reply.body.size());
        if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("models")
            || !doc["models"].IsObject()) {
            error = "unexpected response from /models: " + reply.body.substr(0, 300);
        } else {
            auto text = [](const rapidjson::Value& obj, const char* key) {
                auto f = obj.FindMember(key);
                return f != obj.MemberEnd() && f->value.IsString() ? std::string(f->value.GetString()) : std::string();
            };
            for (auto it = doc["models"].MemberBegin(); it != doc["models"].MemberEnd(); ++it) {
                ModelInfo m;
                m.name = it->name.GetString();
                if (it->value.IsObject()) {
                    if (it->value.HasMember("names"))
                        m.class_names = ParseNames(it->value["names"]);
                    m.file   = text(it->value, "file");
                    m.status = text(it->value, "status");
                }
                m.workers = static_cast<int>(Num(it->value, "workers"));
                models.push_back(std::move(m));
            }
        }
    }

    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->error = error;
    if (error.empty())
        impl_->models = std::move(models);
    return error.empty();
}

std::vector<ModelInfo> Client::Models() const
{
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->models;
}

std::string Client::LastError() const
{
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->error;
}

Result Client::Detect(const std::string& model, const cv::Mat& frame,
                      float conf, float iou, int jpegQuality) const
{
    return impl_->DetectFrame(model, frame, conf, iou, jpegQuality);
}

Result Client::Detect(const std::string& model, const std::vector<uint8_t>& encoded,
                      float conf, float iou) const
{
    return impl_->DetectEncoded(model, encoded.data(), encoded.size(), conf, iou);
}

// The thread keeps its own reference to Impl, so the Client may go away first.
std::future<Result> Client::DetectAsync(const std::string& model, const cv::Mat& frame,
                                        float conf, float iou, int jpegQuality) const
{
    return std::async(std::launch::async, [impl = impl_, model, frame, conf, iou, jpegQuality] {
        return impl->DetectFrame(model, frame, conf, iou, jpegQuality);
    });
}

std::future<Result> Client::DetectAsync(const std::string& model, std::vector<uint8_t> encoded,
                                        float conf, float iou) const
{
    return std::async(std::launch::async,
                      [impl = impl_, model, encoded = std::move(encoded), conf, iou] {
        return impl->DetectEncoded(model, encoded.data(), encoded.size(), conf, iou);
    });
}

// Ultralytics' plot palette (as BGR): a class gets the same colour as in
// Ultralytics' own images, and none of them is black.
static const cv::Scalar kPalette[] = {
    {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255}, {49, 210, 207},
    {10, 249, 72}, {23, 204, 146}, {134, 219, 61}, {52, 147, 26}, {187, 212, 0},
    {168, 153, 44}, {255, 194, 0}, {147, 69, 52}, {255, 115, 100}, {236, 24, 0},
    {255, 56, 132}, {133, 0, 82}, {255, 56, 203}, {200, 149, 255}, {199, 55, 255},
};

void Draw(cv::Mat& frame, const std::vector<Detection>& dets)
{
    Draw(frame, dets, DrawOptions{});
}

void Draw(cv::Mat& frame, const std::vector<Detection>& dets, const DrawOptions& options)
{
    if (frame.empty())
        return;
    const int    thick     = options.thickness > 0 ? options.thickness : std::max(2, frame.cols / 640 * 2);
    const double scale     = options.font_scale > 0 ? options.font_scale : std::max(0.4, frame.cols / 2000.0);
    const int    textThick = std::max(1, thick / 2);
    const int    font      = cv::FONT_HERSHEY_SIMPLEX;
    const int    pad       = 3;
    const cv::Rect bounds(0, 0, frame.cols, frame.rows);

    for (const auto& d : dets) {
        const cv::Scalar& col = kPalette[std::abs(d.class_id) % std::size(kPalette)];
        if (options.boxes)
            cv::rectangle(frame, d.box, col, thick);
        if (!options.labels)
            continue;

        const std::string text = options.scores ? d.label + " " + cv::format("%.2f", d.score) : d.label;
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, font, scale, textThick, &baseline);
        const int tagW = ts.width + 2 * pad;
        const int tagH = ts.height + baseline + 2 * pad;

        // Above the box; below it when there is no room above; inside only as a last resort.
        int y = d.box.y - tagH;
        if (y < 0)
            y = d.box.br().y + tagH <= frame.rows ? d.box.br().y : d.box.y;
        const int x = std::min(std::max(d.box.x, 0), std::max(frame.cols - tagW, 0));
        const cv::Rect tag = cv::Rect(x, y, tagW, tagH) & bounds;
        if (tag.empty())
            continue;

        if (options.tag_alpha >= 1.0) {
            frame(tag).setTo(col);
        } else if (options.tag_alpha > 0.0) {
            cv::Mat roi = frame(tag);
            const cv::Mat fill(roi.size(), roi.type(), col);
            cv::addWeighted(fill, options.tag_alpha, roi, 1.0 - options.tag_alpha, 0.0, roi);
        }
        const cv::Scalar textCol = options.tag_alpha > 0.0 ? cv::Scalar(255, 255, 255) : col;
        cv::putText(frame, text, cv::Point(x + pad, y + pad + ts.height), font, scale, textCol, textThick, cv::LINE_AA);
    }
}

}  // namespace yolohttp
