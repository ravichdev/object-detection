import * as tf from "@tensorflow/tfjs";
import {
  SupportedModels,
  createSegmenter,
  toColoredMask,
  bodyPixMaskValueToRainbowColor,
  drawMask,
  drawPixelatedMask,
} from "@tensorflow-models/body-segmentation";
import * as mpSelfieSegmentation from "@mediapipe/selfie_segmentation";
import { setupCameraApp } from "./camera";

const video = document.querySelector("#video");
const canvas = document.querySelector("#output");
const canvasCtx = canvas.getContext("2d");
let rafId;
const activeEffect = "";

const CANVAS_NAMES = {
  blurred: "blurred",
  blurredMask: "blurred-mask",
  mask: "mask",
  lowresPartMask: "lowres-part-mask",
};
const offScreenCanvases = {};

function createOffScreenCanvas() {
  if (typeof document !== "undefined") {
    return document.createElement("canvas");
  } else if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(0, 0);
  } else {
    throw new Error("Cannot create a canvas in this context");
  }
}

function ensureOffscreenCanvasCreated(id) {
  if (!offScreenCanvases[id]) {
    offScreenCanvases[id] = createOffScreenCanvas();
  }
  return offScreenCanvases[id];
}

function drawAndBlurImageOnCanvas(image, blurAmount, canvas) {
  const { height, width } = image;
  const ctx = canvas.getContext("2d");
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.filter = `blur(${blurAmount}px)`;
  ctx.drawImage(image, 0, 0, width, height);
  ctx.restore();
}

function drawAndBlurImageOnOffScreenCanvas(
  image,
  blurAmount,
  offscreenCanvasName
) {
  const canvas = ensureOffscreenCanvasCreated(offscreenCanvasName);
  if (blurAmount === 0) {
    renderImageToCanvas(image, canvas);
  } else {
    drawAndBlurImageOnCanvas(image, blurAmount, canvas);
  }
  return canvas;
}

function renderImageToCanvas(image, canvas) {
  const { width, height } = image;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  ctx.drawImage(image, 0, 0, width, height);
}

function flipCanvasHorizontal(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
}

function drawWithCompositing(ctx, image, compositeOperation) {
  ctx.globalCompositeOperation = compositeOperation;
  ctx.drawImage(image, 0, 0);
}

async function app() {
  setupCameraApp();

  runBodySegmentation();
}

function onResults(results) {
  // Get Video Properties
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  // Set video width
  video.width = videoWidth;
  video.height = videoHeight;

  // Set canvas height and width
  canvas.width = videoWidth;
  canvas.height = videoHeight;

  // Draw the overlays.
  canvasCtx.save();

  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  drawWithCompositing(canvasCtx, results.segmentationMask, "destination-in");

  const blurredImage = drawAndBlurImageOnOffScreenCanvas(
    results.image,
    15,
    CANVAS_NAMES.blurred
  );

  drawWithCompositing(canvasCtx, blurredImage, "destination-over");

  canvasCtx.restore();
}

const runBodySegmentation = async () => {
  const selfieSegmentation = new mpSelfieSegmentation.SelfieSegmentation({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
    },
  });
  selfieSegmentation.setOptions({
    modelSelection: 1,
  });
  selfieSegmentation.onResults(onResults);

  const renderPrediction = async () => {
    await selfieSegmentation.send({ image: video });

    rafId = requestAnimationFrame(renderPrediction);
  };

  video.addEventListener('loadeddata', async() => {
    await renderPrediction();
  }, false);

};

app();
