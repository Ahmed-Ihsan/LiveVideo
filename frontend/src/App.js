import React, { useRef, useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [name, setName] = useState("");
  const [images, setImages] = useState([]);
  const [streaming, setStreaming] = useState(false);

  const startVideo = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
        })
        .catch((err) => {
          console.error("Error accessing camera: ", err);
        });
    } else {
      console.error("getUserMedia not supported in this browser.");
    }
  };

  const captureImages = async () => {
    setCapturing(true);
    const capturedImages = [];
    for (let count = 0; count < 80; count++) {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      context.drawImage(
        videoRef.current,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const dataUrl = canvas.toDataURL("image/jpeg");
      const blob = await (await fetch(dataUrl)).blob();
      const file = new File([blob], `capture_${count + 1}.jpg`, {
        type: "image/jpeg",
      });
      capturedImages.push(file);

      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    setImages(capturedImages);
    setCapturing(false);
    console.log("Captured 80 images");
  };

  const sendImages = async () => {
    const formData = new FormData();
    formData.append("name", name);
    images.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/add-face",
        formData
      );
      console.log("Images sent successfully: ", response.data);
    } catch (error) {
      console.error("Error sending images: ", error);
    }
  };

  const startRecognition = () => {
    setStreaming(true);
  };

  const trainFaceRecognition = async () => {
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/train-face-recognition"
      );
      console.log("Face recognition training started: ", response.data);
    } catch (error) {
      console.error("Error starting face recognition training: ", error);
    }
  };

  const handleCaptureClick = () => {
    if (!capturing) {
      captureImages();
    }
  };

  const handleSendClick = () => {
    if (images.length === 80) {
      sendImages();
    } else {
      console.error("Not enough images captured.");
    }
  };

  return (
    <div className="app">
      <h1 className="title">Face Recognition</h1>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter your name"
        className="input"
      />
      <button onClick={startVideo} className="button">
        Start Video
      </button>
      {streaming ? (
        <img
          src="http://127.0.0.1:8000/video_feed"
          alt="Video Stream"
          className="stream"
        />
      ):(
        <video ref={videoRef} autoPlay className="video" />
      )}
      <button
        onClick={handleCaptureClick}
        className="button"
        disabled={capturing}
      >
        {capturing ? "Capturing..." : "Capture 80 Images"}
      </button>
      <button
        onClick={handleSendClick}
        className="button"
        disabled={images.length !== 80}
      >
        Send Images
      </button>
      <button onClick={startRecognition} className="button">
        Start Recognition
      </button>
      <button onClick={trainFaceRecognition} className="button">
        Train Face Recognition
      </button>
    </div>
  );
}

export default App;
