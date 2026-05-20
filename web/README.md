# Browser ONNX Demo

Place the selected export here:

```text
leaf_detector/web/models/best.onnx
```

Then serve the folder from a local web server:

```powershell
cd D:\Ken\leaf_detector\web
python -m http.server 8080
```

Open `http://localhost:8080`.

The demo expects a standard Ultralytics YOLO detect ONNX export without built-in
NMS. It performs letterboxing, confidence filtering, and NMS in JavaScript.
