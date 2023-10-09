import * as tf from '@tensorflow/tfjs';
import * as cocossd from "@tensorflow-models/coco-ssd";
import { setupCameraApp } from './camera';

async function app() {
	setupCameraApp();

	runCoco();
}

// Main function
const runCoco = async () => {
    const model = await cocossd.load();
    setInterval(() => {
      detect(model);
    }, 100);
};

const canvas = document.querySelector('#output');
const video = document.querySelector('#video');
const detect = async (net) => {
    // Check data is available
    if (
      video &&
      video.readyState === 4
    ) {
      // Get Video Properties
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      // Set video width
      video.width = videoWidth;
      video.height = videoHeight;

      // Set canvas height and width
      canvas.width = videoWidth;
      canvas.height = videoHeight;

      // Make Detections
      const obj = await net.detect(video);

      // Draw mesh
      const ctx = canvas.getContext("2d");
      drawRect(obj, ctx);
    }
};

const drawRect = (detections, ctx) =>{
	// Loop through each prediction
	detections.forEach(prediction => {

	  // Extract boxes and classes
	  const [x, y, width, height] = prediction['bbox'];
	  const text = prediction['class'];

	  // Set styling
	  const color = '#C70039';
	  ctx.strokeStyle = color
	  ctx.font = '20px Arial';
	  ctx.lineWidth = 2;

	  // Draw rectangles and text
	  ctx.beginPath();
	  ctx.fillStyle = color
	  ctx.fillText(text, x, y);
	  ctx.rect(x, y, width, height);
	  ctx.stroke();
	});
}

app();
