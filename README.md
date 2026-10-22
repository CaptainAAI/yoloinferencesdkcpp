# NewSDKhttp (YoloHttp)

Source of the YoloHttp SDK: a C++ client DLL (libcurl + OpenCV) for the YOLO FastAPI server in `server/`.
How to use the SDK: [`sdk/README.md`](sdk/README.md) (copied into the packed SDK).

| Path | |
| --- | --- |
| `NewSDKhttp/` | DLL project → `bin\x64\<Config>\YoloHttp.dll`. Header in `include/`, code in `src/`, dependency settings in `YoloHttp.Build.props`. |
| `server/server.py` | FastAPI server for the Jetson (`~/yoloinference`), one worker thread per model, with monitoring and model management endpoints. |
| `web/` | npm web dashboard for the Jetson (`~/yoloinferenceweb`): monitoring and model management through `server.py`. |
| `imageserver/` | Image server for the Jetson (`~/yoloimages`, 127.0.0.1:6770): keeps the images of models with *Save images* on. |
| `deploy/` | `deploy_inference.sh`, `deploy_imageserver.sh`: run on the Jetson to install and restart the servers. |
| `sdk/` | Templates for the packed SDK (props, README, notices). |
| `pack_sdk.ps1` | Builds Release x64 and packs `YoloHttp-SDK\` + `YoloHttp-SDK.zip`. |

Dependencies (not in this folder):

- OpenCV 5.0: `..\library\opencv\build\` by default; otherwise `YOLOHTTP_OPENCV_ROOT` or `/p:OpenCVRoot=...`.
- curl + rapidjson from vcpkg (`vcpkg install curl:x64-windows rapidjson:x64-windows`): `D:\vcpkg\installed\x64-windows\`
  by default; otherwise `VCPKG_ROOT`, `YOLOHTTP_CURL_ROOT` or `/p:CurlRoot=...`.

Build and pack:

```powershell
powershell -ExecutionPolicy Bypass -File pack_sdk.ps1
```
