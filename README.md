# yoloinferencesdkcpp — YoloHttp C++ SDK

A C++ library (Windows DLL) for YOLO object detection on a Jetson inference server over HTTP. A program passes a
`cv::Mat` and gets boxes back; JPEG encoding, the HTTP upload (libcurl) and JSON parsing all happen inside
`YoloHttp.dll`. Requests for different models run at the same time, so `DetectAsync` gets three models' answers in
about the time of one.

```cpp
#include "YoloHttp.h"

yolohttp::Client client("140.129.7.180:6767");        // once
yolohttp::Result r = client.Detect("model1", frame);   // per image (cv::Mat, BGR)
if (r.ok)
    yolohttp::Draw(frame, r.detections);               // boxes + labels (DrawOptions: boxes only, no scores, ...)
else
    std::cerr << r.error << "\n";
```

It is one of four repositories that work together:

| Repository | What it is | Runs on | Port |
| --- | --- | --- | --- |
| [yoloinferencemodel](https://github.com/CaptainAAI/yoloinferencemodel) | Inference server `server.py` | Jetson | 6767 |
| [yoloinferencewebserver](https://github.com/CaptainAAI/yoloinferencewebserver) | Web dashboard | Jetson | 8080 |
| [yoloinferenceimageserver](https://github.com/CaptainAAI/yoloinferenceimageserver) | Image store | Jetson, local only | 6770 |
| **yoloinferencesdkcpp** (this one) | C++ SDK `YoloHttp.dll` | Windows PC | none |

## Contents

- [Using the SDK](#using-the-sdk)
- [Examples](#examples)
- [API reference](#api-reference)
- [Behaviour to know](#behaviour-to-know)
- [Building from source](#building-from-source)
- [Repository layout](#repository-layout)
- [How it works inside](#how-it-works-inside)
- [Troubleshooting](#troubleshooting)
- [Third-party code](#third-party-code)

## Using the SDK

### 1. Get the package

Build it with `pack_sdk.ps1` (see [Building from source](#building-from-source)), or take a `YoloHttp-SDK.zip` built
before. It contains:

| Folder | Contents |
| --- | --- |
| `include/` | `YoloHttp.h` and the OpenCV 5.0 headers |
| `lib/` | `YoloHttp.lib`, `opencv_world500.lib` |
| `bin/` | `YoloHttp.dll`, `libcurl.dll`, `z.dll`, `opencv_world500.dll` (all four are needed at run time) |
| `YoloHttp.props` | Visual Studio property sheet |
| `README.md` | This file |
| `licenses/` | Third-party licenses |

### 2. Add it to a Visual Studio project

1. Set the platform to **x64**.
2. View → Other Windows → **Property Manager** → right-click the project → **Add Existing Property Sheet** →
   `YoloHttp-SDK\YoloHttp.props`.
3. `#include "YoloHttp.h"` and build. The four DLLs are copied next to the `.exe` after every build.

The property sheet sets the include and library paths, links `YoloHttp.lib` and `opencv_world500.lib`, and sets
C++17, the `/MD` runtime and an undefined `_DEBUG`. `std::string`, `std::vector`, `std::future` and `cv::Mat` cross
the DLL boundary, so the program must use the DLL's runtime; Debug configurations build too, with the Release
runtime (the D9025 warning about `/U_DEBUG` is expected).

### 3. Have the server running

The Jetson must run [yoloinferencemodel](https://github.com/CaptainAAI/yoloinferencemodel) (port 6767). Check it with
`curl http://140.129.7.180:6767/models`.

## Examples

**One image, one model**

```cpp
#include "YoloHttp.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main()
{
    yolohttp::Client client("140.129.7.180:6767");

    cv::Mat img = cv::imread("test.jpg");
    yolohttp::Result r = client.Detect("model1", img);
    if (!r.ok) {
        std::cerr << "failed: " << r.error << "\n";
        return 1;
    }
    for (const yolohttp::Detection& d : r.detections)
        std::cout << d.label << " " << d.score << " " << d.box << "\n";

    yolohttp::Draw(img, r.detections);
    cv::imwrite("result.jpg", img);
}
```

**Check the server and list its models.** The name is what `Detect` takes; it can differ from the file name.

```cpp
if (!client.Connect()) {
    std::cerr << "Connect failed: " << client.LastError() << "\n";
    return 1;
}
for (const yolohttp::ModelInfo& m : client.Models())
    std::cout << m.name << " -> " << m.file << " (" << m.status << ", " << m.class_names.size() << " classes)\n";
// e.g.  omkegas -> model3.engine (ready, 80 classes)
```

**Send an image file as it is** (no decoding or re-encoding on the PC):

```cpp
std::ifstream f("test.jpg", std::ios::binary);
std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
yolohttp::Result r = client.Detect("model1", bytes);
```

**Several models on the same image at once**

```cpp
auto f1 = client.DetectAsync("model1", img);
auto f2 = client.DetectAsync("model2", img);
auto f3 = client.DetectAsync("particle", img);
yolohttp::Result r1 = f1.get(), r2 = f2.get(), r3 = f3.get();   // about 57 ms instead of 110 ms
```

**Camera or video loop**

```cpp
cv::VideoCapture cap(0);                       // or a video file or an RTSP URL
cv::Mat frame;
while (cap.read(frame)) {
    yolohttp::Result r = client.Detect("model1", frame);
    if (r.ok)
        yolohttp::Draw(frame, r.detections);
    cv::imshow("detect", frame);
    if (cv::waitKey(1) == 27) break;          // Esc
}
```

**Options:** `yolohttp::Client client("140.129.7.180:6767", 3000);` gives up on a request after 3 s;
`client.Detect("model1", img, 0.5f, 0.45f)` keeps only boxes with confidence 0.5 or more;
`client.Detect("model1", img, 0.25f, 0.45f, 80)` sends JPEG quality 80.

**Where the time went**

```cpp
const yolohttp::Timing& t = r.timing_ms;
std::cout << "round trip " << t.round_trip << " ms = server " << t.total_server
          << " (inference " << t.inference_call << ") + network " << t.round_trip - t.total_server << "\n";
```

## API reference

Everything is in namespace `yolohttp`, declared in `NewSDKhttp/include/YoloHttp.h`.

### `Client`

| Member | Description |
| --- | --- |
| `explicit Client(const std::string& url, int timeout_ms = 10000)` | Stores the address; sends nothing and cannot fail. `url` is `"IP:port"`; `http://` in front and a trailing `/` are accepted. `timeout_ms` limits each whole request (connecting: at most 5 s) |
| `bool Connect()` | `GET /models`. `true`: the server answered and `Models()` is filled. `false`: `LastError()` says why. Optional |
| `std::vector<ModelInfo> Models() const` | The models from the last successful `Connect()` |
| `std::string LastError() const` | The last `Connect()` error; empty after success |
| `const std::string& Url() const` | The address after cleanup, e.g. `http://140.129.7.180:6767` |
| `Result Detect(model, const cv::Mat& frame, float conf = 0.25f, float iou = 0.45f, int jpegQuality = 95) const` | JPEG-encodes the BGR image, sends it to `model`, waits for the answer |
| `Result Detect(model, const std::vector<uint8_t>& encoded, float conf = 0.25f, float iou = 0.45f) const` | Sends bytes that are already an image file |
| `std::future<Result> DetectAsync(...)` | Same arguments; returns at once, the request runs on its own thread, `.get()` returns the `Result` |

### `Draw`

`void Draw(cv::Mat& frame, const std::vector<Detection>& dets)` draws each box and a `label score` tag onto the image,
in place. Each class has its own colour, from Ultralytics' plot palette (the dashboard uses the same colours). A tag
sits above its box, or below it when the box touches the top edge, so
it does not cover the object, and its background is see-through.

`void Draw(cv::Mat& frame, const std::vector<Detection>& dets, const DrawOptions& options)` does the same with your
choices:

| `DrawOptions` field | Default | Meaning |
| --- | --- | --- |
| `bool boxes` | `true` | Draw the box rectangles |
| `bool labels` | `true` | Draw the class-name tag next to each box |
| `bool scores` | `true` | Add the score to the tag (`Particle 0.58`) |
| `double tag_alpha` | `0.6` | Tag background opacity: 1 = solid, 0 = text only |
| `int thickness` | `0` | Box line width in pixels; 0 = from the image width (2 px per 640 px) |
| `double font_scale` | `0` | Tag text size (OpenCV font scale); 0 = from the image width |

```cpp
yolohttp::DrawOptions opt;
opt.labels = false;                         // boxes only: nothing covers the area around the objects
yolohttp::Draw(frame, r.detections, opt);
```

### Structs

| Struct | Field | Meaning |
| --- | --- | --- |
| `Detection` | `cv::Rect box` | Box in pixels of the image passed to `Detect` |
| | `float score` | Confidence, 0 to 1 |
| | `int class_id` | Class index from the model |
| | `std::string label` | Class name from the model, or the index as text |
| `Result` | `bool ok` | `false`: see `error` |
| | `std::string error` | e.g. `HTTP 404: Unknown model 'model9'. Choose from: [...]`, or curl's message |
| | `int http_status` | 0 when the server did not answer at all |
| | `std::string model` | The model the request used |
| | `std::vector<Detection> detections` | The boxes found |
| | `Timing timing_ms` | Where the time went |
| `Timing` (ms) | `encode`, `round_trip` | Measured on the PC: JPEG encoding (0 for bytes), request sent to answer received |
| | `total_server`, `upload_read`, `queue_wait`, `decode` | From the server: the whole request, reading the upload, waiting for the model's thread, decoding |
| | `inference_call`, `preprocess`, `inference`, `postprocess`, `serialization` | From the server: `model.predict()`, its three YOLO stages, turning boxes into a list |
| `ModelInfo` | `std::string name` | The name to pass to `Detect` |
| | `std::string file` | The model file behind it, e.g. `model3.engine` |
| | `std::string status` | `ready`, `loading` or `error`; `Detect` works only when `ready` |
| | `std::map<int, std::string> class_names` | Class id to name |
| | `int workers` | Server threads for this model |

## Behaviour to know

- **Thread safe.** One `Client` may be used from any number of threads at once. Each call returns its own
  `Result`, so errors and timings never mix. Create the `Client` once and keep it: it reuses HTTP connections.
- **Empty result ≠ error.** `r.ok && r.detections.empty()` means no objects; check `ok` first.
- **`DetectAsync` shares the frame.** Don't draw into it or overwrite it until `.get()` returns; pass
  `img.clone()` if you need to. The `Client` may be destroyed before the futures are ready.
- **Image size.** Any size is fine. The server resizes to the model's input size (e.g. 640 × 640) and returns boxes
  in your image's pixels.
- **Network.** Over Ethernet a request takes about 24 ms (server 20 ms); over Tailscale / Wi-Fi about 1 s. Use the
  Jetson's Ethernet address.
- **C++/CLI (WinForms).** `YoloHttp.h` includes `<future>`, which Visual C++ refuses under `/clr`. Include the header
  only in native `.cpp` files (Common Language RunTime Support: No) and call them from the form.

## Building from source

**Requirements:** Visual Studio 2022 (v143, C++17), Windows 10/11 x64, and:

- **OpenCV 5.0** (Windows build folder with `include\opencv2` and `x64\vc16\lib\opencv_world500.lib`). Default:
  `..\library\opencv\build\` next to this folder; otherwise set `YOLOHTTP_OPENCV_ROOT` or pass `/p:OpenCVRoot=...`.
- **libcurl and RapidJSON** from vcpkg: `vcpkg install curl:x64-windows rapidjson:x64-windows`. Default:
  `D:\vcpkg\installed\x64-windows\`; otherwise `VCPKG_ROOT`, `YOLOHTTP_CURL_ROOT` or `/p:CurlRoot=...`.

**Build:** open `NewSDKhttp.sln`, choose Release | x64, Build. Output: `bin\x64\Release\YoloHttp.dll` and
`YoloHttp.lib`, with `libcurl.dll`, `z.dll` and `opencv_world500.dll` copied next to them. A missing dependency stops
the build with a message saying what to set.

**Pack:**

```powershell
powershell -ExecutionPolicy Bypass -File pack_sdk.ps1
```

builds Release x64 and writes `YoloHttp-SDK\` and `YoloHttp-SDK.zip` (about 30 MB).

## Repository layout

| Path | Contents |
| --- | --- |
| `NewSDKhttp.sln` | Visual Studio solution |
| `NewSDKhttp/NewSDKhttp.vcxproj` | The DLL project (target `YoloHttp`) |
| `NewSDKhttp/YoloHttp.Build.props` | Dependencies, defines, runtime, DLL copying |
| `NewSDKhttp/include/YoloHttp.h` | Public API |
| `NewSDKhttp/src/YoloHttp.cpp` | Implementation |
| `sdk/YoloHttp.props` | Property sheet for programs that use the package |
| `sdk/THIRD_PARTY_NOTICES.txt` | Third-party notices for the package |
| `pack_sdk.ps1` | Build and pack |

## How it works inside

1. `Detect(model, frame)` JPEG-encodes the image with OpenCV (`cv::imencode`).
2. A libcurl handle is taken from a small pool (created on demand, returned after use, so its HTTP connection stays
   open for the next request).
3. The image goes to `POST /predict` as a multipart form (`model_name`, `conf`, `iou`, `file`).
4. The JSON answer is parsed with RapidJSON into `Detection` boxes and `Timing`; HTTP errors become
   `HTTP <code>: <detail>`.

`DetectAsync` runs the same on a `std::async` thread, holding a `std::shared_ptr` to the client's state so the
`Client` may go away first. `curl_global_init` runs once per process. No proxy is used (`CURLOPT_PROXY ""`), and the
`Expect: 100-continue` round trip is disabled.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Failed to connect to ... Could not connect to server` | The server is not running or the address is wrong: `curl http://IP:6767/models` |
| `Operation timed out` | Raise `timeout_ms`; use the Ethernet address |
| `HTTP 404: Unknown model ...` | Use a name from `client.Models()` (the name, not the file name) |
| `HTTP 503: Model '...' is loading` | Wait until it is `ready`; `failed to load`: check the dashboard |
| `HTTP 400: Uploaded file is not a decodable image` | The bytes passed to `Detect` are not an image |
| Program does not start: `YoloHttp.dll` / `libcurl.dll` not found | The DLLs from `bin\` must be next to the `.exe` (`YoloHttp.props` copies them) |
| Crash inside `std::vector` or `std::string` | The project uses `/MDd` or defines `_DEBUG`; use `YoloHttp.props` |
| `error C1189: <future> is not supported when compiling with /clr` | Include `YoloHttp.h` only in native `.cpp` files |

## Third-party code

`YoloHttp.dll` compiles in RapidJSON (MIT) and loads libcurl (curl license), zlib (zlib license) and OpenCV 5.0
(Apache 2.0); their licenses are in the package's `licenses/` folder.
